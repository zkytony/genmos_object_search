## A command line interface to run end-to-end
import torch
import os, sys
os.environ["SPACY_WARNING_IGNORE"] = "W008"

import spacy
import json
import random
from pprint import pprint

from sloop.osm.datasets import FILEPATHS, MapInfoDataset

from sloop_object_search.oopomdp.experiments.trial import\
    create_world, make_trial, get_prior, obj_letter_map, obj_id_map
import sloop_object_search.oopomdp.problem as mos
from sloop_object_search.oopomdp.env.env import interpret as interpret_env
from sloop_object_search.oopomdp.env.env import make_laser_sensor, equip_sensors
from sloop_object_search.oopomdp.models.transition_model import RobotTransitionModel
from sloop_object_search.models.heuristics.rules import BASIC_RULES, ForefRule
from sloop_object_search.models.heuristics.model import KeywordModel, RuleBasedModel, MixtureSLUModel


def input_default(prompt, default_val):
    val = input("%s [%s]: " % (prompt, str(default_val)))
    if len(val.strip()) == 0:
        val = default_val
    return val

def load_files(sp="en_core_web_md"):
    print("Loading spacy model...")
    spacy_model = spacy.load(sp)

    print("Loading spatial keywords...")
    with open(FILEPATHS["relation_keywords"]) as f:
        spatial_keywords = json.load(f)

    print("Loading symbol to synonyms...")
    with open(FILEPATHS["symbol_to_synonyms"]) as f:
        symbol_to_synonyms = json.load(f)

    return {"spacy_model": spacy_model,
            "spatial_keywords": spatial_keywords,
            "symbol_to_synonyms": symbol_to_synonyms}

def run_with_hint(map_name, worldstr, hint, mapinfo,
                  sensor, prior_type,
                  landmark_heights=None, flight_height=2,
                  **kwargs):
    # Make prior
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if prior_type == "keyword":
        keyword_model = KeywordModel()
        prior, metadata = get_prior("keyword", keyword_model, hint,
                                    map_name, mapinfo, **kwargs)
        model_name = "keyword"

    elif prior_type == "rule" or prior_type == "mixture":
        basic_rules = BASIC_RULES
        relative_rules = {}
        relative_predicates = {"front", "behind", "left", "right"}
        foref_models = {}
        model_name = "ego_ctx_foref_angle"
        iteration = 2
        for predicate in relative_predicates:
            if predicate in {"front", "behind"}:
                model_keyword = "front"
            else:
                model_keyword = "left"
            foref_model_path = os.path.join(
                "..", "models",
                "iter%s_%s:%s:%s"
                % (iteration, model_name.replace("_", "-"),
                   model_keyword, map_name.replace("_", ",")),
                "%s_model.pt" % model_keyword)
            if os.path.exists(foref_model_path):
                print("Loading %s model for %s" % (model_name, predicate))
                nn_model = torch.load(foref_model_path, map_location=device)
                foref_models[predicate] = nn_model.predict_foref
                relative_rules[predicate] = ForefRule(predicate)
            else:
                import pdb; pdb.set_trace()
                raise ValueError("Pytorch model [%s] for %s does not exist!" % (model_name, predicate))
        rules = {**basic_rules,
                 **relative_rules}
        print("All Rules (%s):" % (model_name))
        pprint(list(rules.keys()))

        if prior_type == "rule":
            model = RuleBasedModel(rules)
        else:
            model = MixtureSLUModel(rules)
        prior, metadata = get_prior(prior_type, model,
                                    hint, map_name, mapinfo,
                                    foref_models=foref_models,
                                    foref_kwargs={"device": device,
                                                  "mapsize": (28,28)},
                                    **kwargs)

    else:
        if prior_type not in {"informed", "uniform"}:
            raise ValueError("Invalid Prior Type %s" % prior_type)
        prior, metadata = prior_type, {}
        model_name = "na"

    # POMDP parameters
    params = {
        "reward_small": 10,
        "no_look": True,
        "exploration_const": 1000,
        "max_time": 360,
        "max_steps": 100,
        "max_depths": 30,
        "num_sims": 300,
        "visualize": True,
        "obj_id_map": obj_id_map
    }

    trial = make_trial("interface-%s" % prior_type,
                       worldstr, map_name, hint, sensor,
                       prior_type,
                       prior,
                       metadata,
                       model_name,
                       **params)
    trial.verbose = True

    if landmark_heights is not None:
        RobotTransitionModel.MAPINFO = mapinfo
        RobotTransitionModel.MAPNAME = map_name
        RobotTransitionModel.FLIGHT_HEIGHT = flight_height
        RobotTransitionModel.LANDMARK_HEIGHTS = landmark_heights
    results = trial.run(logging=True)
    return results

def input_problem_config(loaded_things):
    map_name = input("map name: ")
    if len(map_name.strip()) == 0:
        return None

    print("Loading mapinfo")
    mapinfo = MapInfoDataset()
    mapinfo.load_by_name(map_name)

    # Create a random POMDP domain on the map
    nobj = min(3, int(input_default("num objects, max 3", 2)))
    sensor_range = int(input_default("Sensor range", 4))
    dims = mapinfo.map_dims(map_name)
    all_locs = {(x,y)
                for x in range(dims[0])
                for y in range(dims[1])}
    target_objects = ["R", "G", "B"]  # a list of upper case letters
    target_poses = []  # a list of (x,y) poses
    for i in range(nobj):
        obj_letter = target_objects[i]
        obj_loc = input_default("x, y for object %s" % obj_letter, "random")
        if obj_loc == "random":
            obj_loc = random.sample(all_locs - set(target_poses),1)[0]
        else:
            obj_loc = tuple(map(int, obj_loc.split(",")))
        target_poses.append(obj_loc)

    robot_pose = (15, 20)
    worldstr = create_world(dims[0], dims[1],
                            robot_pose, set({}),
                            target_poses,
                            target_objects)
    sensor = make_laser_sensor(90, (1, sensor_range), 0.5, False)
    worldstr = equip_sensors(worldstr, {'r':sensor})

    # Display the domain on the map
    ### Make a temporary environment
    env = interpret_env(worldstr, no_look=True,
                        obj_id_map=obj_id_map)
    bg_path = FILEPATHS[map_name]["map_png"]
    viz = mos.MosViz(env, controllable=False, bg_path=bg_path, res=20)
    if viz.on_init() == False:
        raise Exception("Environment failed to initialize")
    viz.update('r', None, None, None, {})
    viz.on_render()

    # Get inputs from user
    hint = input("Hint: ")
    return [map_name, mapinfo, robot_pose, worldstr, sensor, hint]


if __name__ == "__main__":
    default_prior = "mixture"
    loaded_things = load_files()

    first_time = True
    rerun_previous = False
    while True:
        if not first_time:
            rerun_previous = input_default("Rerun previous?", "N").lower().startswith("y")

        if not rerun_previous:
            map_name, mapinfo, robot_pose, worldstr, sensor, hint\
                = input_problem_config(loaded_things)

        prior_type = input_default("Prior type", default_prior)
        run_with_hint(map_name, worldstr, hint, mapinfo,
                      sensor, prior_type, landmark_heights={},
                      **loaded_things)
        first_time = False

        print("Done!\n")
