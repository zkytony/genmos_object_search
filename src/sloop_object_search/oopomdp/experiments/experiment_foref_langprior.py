import os, sys
os.environ["SPACY_WARNING_IGNORE"] = "W008"
from sloop_object_search.parsing.parser import parse, match_spatial_keyword
from sloop_object_search.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop_object_search.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop_object_search.models.heuristics.model import KeywordModel, RuleBasedModel, GaussianPointModel, MixtureSLUModel
from sloop_object_search.models.heuristics.rules import BASIC_RULES, ForefRule
from sloop_object_search.oopomdp.env.env import *
from sloop_object_search.oopomdp.experiments.trial import *
from sloop_object_search.oopomdp.experiments.constants import *
import spacy
import json
import copy
import math
import yaml
import random
import pickle
import numpy as np
import torch
from pprint import pprint

# PREDICATES = {"front"} #"near"}#at", "beyond", "between", "north", "west", "south", "east", "on"}

ITERATION = "2"
MODEL_NAME = "ego_ctx_foref_angle"
ALL_MAPS = {"cleveland", "denver", "austin", "honolulu", "washington_dc"}
dataset_path = "../../datasets/SL-OSM-Dataset/"

def load_sgfile(filepath):
    """load the .sg file (spatial language graph)"""
    try:
        with open(filepath) as f:
            sample = json.load(f)
    except UnicodeDecodeError:
        print("Failed to read", filepath)
        return None
    except Exception as ex:
        raise ex
    return sample

def get_targets_info(sample, mapinfo, not_exist_idx=-1):
    """Returns a map from obj letter (R, G, or B) to pomdp location (x, y in
    POMDP grid world)"""
    obj_names = []  # e.g. gcar, bike, rcar
    if "obj1_name" in sample:
        obj_names.append(sample["obj1_name"])
    if "obj2_name" in sample:
        obj_names.append(sample["obj2_name"])

    targets = []
    targets_loc_pomdp = []
    for obj_short_name in obj_names:
        letter = obj_letter_map[obj_short_name]
        loc_idx = sample["%s_loc" % obj_short_name]  # the number corresponding to the grid cell
        if loc_idx != not_exist_idx:
            pomdp_loc = mapinfo.idx_to_pomdp(sample["map_name"], int(loc_idx))
            targets.append(letter)
            targets_loc_pomdp.append(pomdp_loc)

    return targets, targets_loc_pomdp

def build_mapinfo(test_map, all_maps):
    """
    Args:
        test_map: e.g. austin
        all_maps: all city map names
    Returns:
        mapinfo
    """
    mapinfo = MapInfoDataset()
    print("Test map: %s" % test_map)
    mapinfo.load_by_name(test_map)
    print("Train maps:")
    for other_map in all_maps:
        if other_map != test_map:
            # maps[other_map] = num_trials_per_map
            print("  %s" % other_map)
            mapinfo.load_by_name(other_map)
    return mapinfo

def accept_case(d, target_objects,
                all_in=False, spacy_model=None, spatial_keywords=None,
                min_num_rels=1, must_contain=None):
    if len(d["relations"]) < min_num_rels:
        return False
    has_predicate = False
    for rel in d["relations"]:
        best_match, all_matches = match_spatial_keyword(rel[2],
                                                        spatial_keywords,
                                                        spacy_model=spacy_model,
                                                        similarity_thres=0.98,
                                                        match_multiple=True)
        if best_match is None:
            # A relation has no keyword.
            return False

        if must_contain is not None:
            if best_match in predicates:
                has_predicate = True

        if all_in:
            for keyword in all_matches:
                if keyword not in predicates:
                    return False

        mentioned_obj = obj_letter_map[rel[0]]
        if mentioned_obj not in target_objects:
            # The object that is mentioned in the speech
            # is not actually what is on the map.
            return False

        mentioned_landmark_symbol = rel[1]
        if mentioned_landmark_symbol.lower().startswith("robot"):
            # Skip cases where the robot is used as the reference
            return False

    if must_contain is None:
        return True
    else:
        return has_predicate


def get_id(obj_symbol):
    return obj_id_map[obj_letter_map[obj_symbol]]

def make_trial(trial_name, worldstr, map_name, language, sensor, prior_type,
               prior, prior_metadata, model_name, **kwargs):
    problem_args = {"sigma": kwargs.get("sigma", 0.01),
                    "epsilon": kwargs.get("epsilon", 1.0),
                    "agent_has_map": kwargs.get("agent_has_map", True),
                    "reward_small": kwargs.get("small", 10),
                    "sensors": {"r": sensor},
                    "no_look": kwargs.get("no_look", True)}
    solver_args = {"max_depth": kwargs.get("max_depth", 30),
                   "discount_factor": kwargs.get("discount_factor", 0.95),
                   "planning_time": kwargs.get("planning_time", -1.),
                   "num_sims": kwargs.get("num_sims", -1),
                   "exploration_const": kwargs.get("exploration_const", 1000)}
    exec_args = {"max_time": kwargs.get("max_time", 360),
                 "max_steps": kwargs.get("max_steps", 200),
                 "visualize": kwargs.get("visualize", False)}
    sensor_str = sensor.replace(" ", ":").replace("_", "*")
    return SloopPriorTrial("%s_%s-%s-%s-%s" % (trial_name, prior_type,
                                               map_name.replace("_",","),
                                               sensor_str, model_name.replace("_",">")),
                           config={"problem_args": problem_args,
                                   "solver_args": solver_args,
                                   "exec_args": exec_args,
                                   "world": worldstr,
                                   "map_name": map_name,
                                   "prior_type": prior_type,
                                   "prior": prior,
                                   "prior_metadata": prior_metadata,
                                   "language": language,
                                   "obj_id_map": kwargs.get("obj_id_map", {})})

def create_world(width, length, robot_pose, landmark_poses, target_poses, target_objects):
    worldstr = [[ "." for i in range(width)] for j in range(length)]

    rx, ry = robot_pose
    worldstr[ry][rx] = "r"

    for i, pose in enumerate(target_poses):
        x, y = pose
        assert worldstr[y][x] == ".", "Invalid landmark pose %s" % str((x,y))
        worldstr[y][x] = target_objects[i]  # target_objects[i] is a character

    for x,y in landmark_poses:
        assert worldstr[y][x] == ".", "Invalid landmark pose %s" % str((x,y))
        worldstr[y][x] = "x"

    # Create the string.
    finalstr = []
    for row_chars in worldstr:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)
    return finalstr


def get_prior(prior_type, model, query, map_name, mapinfo,
              **kwargs):

    metadata = {}
    if prior_type == "uniform":
        prior = "uniform"

    elif prior_type == "informed":
        prior = "informed"

    elif prior_type.startswith("informed"):
        obj_poses = kwargs.get("obj_poses")
        prior = model.interpret(obj_poses, map_name, mapinfo)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}

    elif prior_type.startswith("keyword"):
        prior, meta = model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    elif prior_type.startswith("rule"):
        prior, meta =\
            model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    elif prior_type.startswith("mixture"):
        prior, meta =\
            model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    return prior, metadata

# For now, to understand exact performance benefit, just pick samples with
# the given keyword
def main(test_map_name, model_name, spacy_model,
         sgdir, robot_poses_yaml_file,
         relative_predicates={"front", "behind", "left", "right"},
         map_dims=(41,41),
         use_annotated=False,
         search_one_target=True,
         baselines={"informed", "keyword", "rule_based", "uniform"},
         mixture_specs={"basic": ([0, 3], [0.8, 0.2]),
                        "full": ([0, 1, 2, 3], [0.6, 0.25, 0.1, 0.05])}):

    if map_dims[0] == 41:
        not_exist_idx = -1
    elif map_dims[0] == 21:
        raise ValueError("21x21 maps not allowed.")

    mapinfo = build_mapinfo(test_map_name, ALL_MAPS)

    print("Loading spatial keywords...")
    with open(FILEPATHS["relation_keywords"]) as f:
        spatial_keywords = json.load(f)

    print("Loading symbol to synonyms...")
    with open(FILEPATHS["symbol_to_synonyms"]) as f:
        symbol_to_synonyms = json.load(f)

    ########### building models ###############

    # Rule based models
    if "rule_based" in baselines or "mixture" in baselines:
        print("Building rule based models...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        foref_models = {}  # map from predicate to nn model
        mixture_models = {}  # map from mixture type to
        rulebased_model = None  # rule based model

        basic_rules = BASIC_RULES

        iteration, model_name = model_name

        relative_rules={}
        for predicate in relative_predicates:
            if predicate in {"front", "behind"}:
                model_keyword = "front"
            elif predicate in {"left", "right"}:
                model_keyword = "left"
            else:
                raise ValueError("currently Foref prediction only for left/right/front/behind")

            foref_model_path = os.path.join(
                "resources", "models",
                "iter%s_%s:%s:%s"
                % (iteration, model_name.replace("_", "-"),
                   model_keyword, test_map_name.replace("_", ",")),
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
        pprint(rules)
        rulebased_model = RuleBasedModel(rules)

        for mixture_type in mixture_specs:
            mixture_idxes, mixture_weights = mixture_specs[mixture_type]
            mixture_models[mixture_type] = MixtureSLUModel(rules, mixture_idxes, mixture_weights)

    # Keyword model
    if "keyword" in baselines:
        print("Building keyword model...")
        keyword_model = KeywordModel()

    # Gaussian point model
    infgaus_models = {}  # maps from model name to model
    for b in baselines:
        if b.startswith("informed") and b != "informed":
            sigma = int(b.split("#")[1])
            print("Building Informed model with sigma %d..." % sigma)
            infgaus_models[sigma] = GaussianPointModel(sigma)



    ########### Loading sg files into cases ###############

    print("Loading data into cases...")
    counts = {}
    data = []
    tested_langauges = set()
    cases = []
    total_skipped = 0

    # Use the annotated language for comparison; (but may not use the annotated relations)

    # ASSUME sg_map_dir is organized as {test_map}/sg-{test_map}-{seed}.json
    # and ASSUME that each case, i.e. (test_map, seed) ALREADY has a robot pose
    # chosen, that is stored in the `robot_poses_yaml_file`.
    sg_map_dir = os.path.join(sgdir, test_map_name)
    with open(robot_poses_yaml_file) as f:
        robot_poses = yaml.load(f, Loader=yaml.Loader)

    cases = []
    for sg_filename in sorted(os.listdir(sg_map_dir)):
        map_name, seed = os.path.splitext(sg_filename)[0].split("-")[1:]
        assert map_name == test_map_name

        robot_pose = robot_poses[(map_name, seed)][:2]

        sample = load_sgfile(os.path.join(sg_map_dir, sg_filename))
        targets, targets_loc_pomdp = get_targets_info(sample, mapinfo, not_exist_idx)
        symbol_to_poses = {letter_symbol_map[targets[i]]: targets_loc_pomdp[i]
                           for i in range(len(targets))}

        w, l = map_dims

        if search_one_target:
            # Search one target at a time
            for i in range(len(targets)):
                worldstr = create_world(w, l, robot_pose, set({}),
                                        [targets_loc_pomdp[i]], [targets[i]])
                seed_i = "{:02d}".format(int(seed) + i)
                case = (worldstr, map_name, seed_i, sample, symbol_to_poses)
                cases.append(case)
        else:
            worldstr = create_world(w, l, robot_pose, set({}),
                                    targets_loc_pomdp, targets)
            case = (worldstr, map_name, seed, sample, symbol_to_poses)
            cases.append(case)

        hint = sample["lang_original"]

    ########################## create trials #######################

    # Create trials
    all_trials = []
    sensor_ranges = [3,4,5]
    for i, case in enumerate(cases):
        print("case %d" % i)
        params = {
            "reward_small": 10,
            "no_look": True,
            "exploration_const": 1000,
            "max_time": 1e9,
            "max_steps": 200,
            "max_depths": 30,
            "visualize": False,
            "num_sims": 1000,
            "obj_id_map": obj_id_map
        }

        worldstr, map_name, seed, sample, obj_poses = case
        language = sample["lang_original"]
        if use_annotated:
            query = sample  # The whole thing is the sg_dict (it has "relations")
            auto_parse = ""
        else:
            query = language
            auto_parse = "#auto"  # suffix #auto to prior type (e.g. keyword#auto)
        print(query)

        # two gaussians informed baselines
        infgaus_priors = {}
        if len(infgaus_models) > 0:
            for sigma in infgaus_models:
                infgaus_prior, _ = get_prior("informed-%d" % sigma, infgaus_models[sigma], None, map_name,
                                             mapinfo, obj_poses=obj_poses)
                infgaus_priors[sigma] = infgaus_prior

        if "keyword" in baselines:
            kw_model_allowed_keywords = relative_predicates
            kw_model_allowed_keywords |= set(basic_rules.keys())
            keyword_prior, kw_metadata =\
                get_prior("keyword", keyword_model, query, map_name, mapinfo,
                          spatial_keywords=spatial_keywords,
                          symbol_to_synonyms=symbol_to_synonyms,
                          spacy_model=spacy_model,
                          allowed_keywords=kw_model_allowed_keywords)

        rb_priors, rb_metas = {}, {}
        if "rule_based" in baselines:
            foref_kwargs={"device": device,
                          "mapsize": (28,28)}
            rulebased_prior, rule_metadata =\
                get_prior("rule_based", rulebased_model,
                          query, map_name, mapinfo,
                          symbol_to_synonyms=symbol_to_synonyms,
                          spatial_keywords=spatial_keywords,
                          foref_models=foref_models,
                          foref_kwargs=foref_kwargs,
                          spacy_model=spacy_model)
            rb_priors[model_name] = rulebased_prior
            rb_metas[model_name] = rule_metadata

        mx_priors, mx_metas = {}, {}
        if "mixture" in baselines:
            for mixture_type in mixture_models:
                mixture_model = mixture_models[mixture_type]
                mixture_prior, mixture_meta =\
                    get_prior("mixture", mixture_model,
                              query, map_name, mapinfo,
                              symbol_to_synonyms=symbol_to_synonyms,
                              spatial_keywords=spatial_keywords,
                              foref_models=foref_models,
                              foref_kwargs=foref_kwargs,
                              spacy_model=spacy_model)
                mx_priors[mixture_type] = mixture_prior
                mx_metas[mixture_type] = mixture_meta

        trial_name = "langprior-{}_{}".format(map_name.replace("_", ","), seed)
        for sensor_range in sensor_ranges:
            sensor = make_laser_sensor(90, (1, sensor_range), 0.5, False)

            if "uniform" in baselines:
                uniform_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                           "uniform", "uniform", {}, "na", **params)
                all_trials.append(uniform_trial)

            if "informed" in baselines:
                informed_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                            "informed", "informed", {}, "na", **params)
                all_trials.append(informed_trial)

            for sigma in infgaus_models:
                infgaus_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                           "informed#%d" % sigma, infgaus_priors[sigma], {}, "na", **params)
                all_trials.append(infgaus_trial)


            if "keyword" in baselines:
                keyw_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                        "keyword{}".format(auto_parse), keyword_prior,
                                        kw_metadata, "na", **params)
                all_trials.append(keyw_trial)

            if len(rb_priors) > 0:
                rule_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                        "rule#based#{}{}".format(model_name.replace("_",">"), auto_parse),
                                        rb_priors[model_name],
                                        rb_metas[model_name],
                                        model_name, **params)
                all_trials.append(rule_trial)

            for mixture_type in mixture_models:
                mixture_trial = make_trial(trial_name, worldstr, map_name, language, sensor,
                                           "mixture#{}{}".format(mixture_type, auto_parse),
                                           mx_priors[mixture_type],
                                           mx_metas[mixture_type],
                                           model_name, **params)
                all_trials.append(mixture_trial)

    # Generate scripts to run experiments and gather results
    return all_trials


if __name__ == "__main__":
    # Loading resources
    print("Loading spacy model...")

    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join("results", "march31-2021")
    sgdir = "./fixes/march30_2021-languages-keyword#auto"
    robot_poses_yaml_file = "./fixes/march30_2021-robot_poses/robot_poses.yaml"

    # Some experiment-level configs
    use_annotated = True  # use annotated language
    search_one_target = True  # search one target at a time
    baselines = {#"informed",
        "keyword",
        "rule_based",
        "mixture"}
                 #"uniform",
                 #"informed#5",
                 #"informed#15"}
    spacy_model = spacy.load("en_core_web_md")
    model_name = (2, "ego_ctx_foref_angle")
    all_trials = []
    for map_name in sorted(["austin", "denver", "cleveland", "honolulu", "washington_dc"]):
        map_trials = main(map_name, model_name, spacy_model,
                          sgdir, robot_poses_yaml_file,
                          relative_predicates={"front", "behind", "left", "right"},
                          use_annotated=use_annotated,
                          search_one_target=search_one_target,
                          baselines=baselines,
                          mixture_specs={"basic": ([0, 3], [0.8, 0.2]),
                                         "full": ([0, 1, 2, 3], [0.6, 0.25, 0.1, 0.05])})
        all_trials.extend(map_trials)

    if len(all_trials) > 0:
        random.shuffle(all_trials)
        annotated = "-annotated" if use_annotated else ""
        search_one = "-onetarget" if search_one_target else ""
        exp = Experiment("NewEvalpomdpBB-%s%s" % (annotated, search_one),
                         all_trials, output_dir, verbose=True,
                         add_timestamp=False)
        exp.generate_trial_scripts(split=400)
        print("Find multiple computers to run these experiments.")
    else:
        print("NO TRIALS.")
