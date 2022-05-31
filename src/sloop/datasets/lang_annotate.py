# Given a path to the sg_parsed directory,
# accept/reject each. If reject can provide
# your own spatial relations.


import os
import argparse
from sloop.utils import print_info, print_error, print_warning, print_success, \
    print_info_bold, print_error_bold, print_warning_bold, print_success_bold
import json
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
import random
import copy
from difflib import SequenceMatcher


TARGETS = {"RedHonda", "RedBike", "GreenToyota"}

def input_default(prompt, default_val):
    val = input("%s [%s]: " % (prompt, str(default_val)))
    if len(val.strip()) == 0:
        val = default_val
    return val

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def suggestions(landmarks, user_input):
    res = []
    for landmark in landmarks:
        if similar(landmark, user_input) >= 0.7:
            res.append(landmark)
    return list(sorted(res))

# User interactively annotates an sg_dict
def do_annotation(sg_dict, mapinfo, spatial_keywords):
    entities = set()
    relations = []
    while True:
        print_warning("Enter relation:")
        relstr = input()
        if len(relstr.strip()) == 0:
            skip = input_default("End?", "y")
            if skip == "y":
                break
        else:
            if len(relstr.split(",")) != 3:
                print_error("Error: relation must be TargetSymbol, relation, LandmarkSymbol")
                continue
            target, relation, landmark = relstr.split(",")
            target = target.strip()
            relation = relation.strip()
            landmark = landmark.strip()
            if target not in TARGETS:
                print_error("Error: target %s is not recognized." % target)
                continue
            valid_landmarks = mapinfo.landmarks_for(sg_dict["map_name"])
            if landmark not in valid_landmarks:
                print_error("Error: landmark %s is not recognized in %s." % (landmark, sg_dict["map_name"]))
                print_error("    Suggestions: %s" % str(suggestions(valid_landmarks, landmark)))
                continue
            if relation not in spatial_keywords:
                print_error("Error: keyword %s is not recognized." % relation)
                print_error("    Suggestions: %s" % str(suggestions(spatial_keywords, relation)))
                continue
            relations.append((target, landmark, relation))
            entities.add(target)
            entities.add(landmark)
    if len(relations) == 0:
        return None
    entities = list(sorted(entities))
    sg_ann = copy.deepcopy(sg_dict)
    sg_ann["entities"] = entities
    sg_ann["relations"] = relations
    sg_ann["manual"] = True
    return sg_ann

def print_intro():
    print("\nHere is how you annotate a relation. When being prompted, you enter the following")
    print_warning("Enter relation:")
    print("TargetSymbol, relation, LandmarkSymbol\n")

def print_sg_dict(sg_dict):
    print_info_bold(sg_dict["lang_original"])
    print_warning("Entities:")
    print_info("    " + str(sg_dict["entities"]))
    print_warning("Relations:")
    for rel in sg_dict["relations"]:
        print_info("    " + "(%s, %s, %s)" % (rel[0], rel[2], rel[1]))

def print_sg_ann(sg_ann):
    print_warning("Entities:")
    print_success("    " + str(sg_ann["entities"]))
    print_warning("Relations:")
    for rel in sg_ann["relations"]:
        print_success("    " + "(%s, %s, %s)" % (rel[0], rel[2], rel[1]))


def main():
    parser = argparse.ArgumentParser(description="Annotate languages")
    parser.add_argument("sg_dir", type=str, help="Path to sg dict json files")
    parser.add_argument("map_name", type=str, help="map name you care about for this round.")
    parser.add_argument("outdir", type=str, help="Output directory")
    args = parser.parse_args()

    print_intro()

    mapinfo = MapInfoDataset()
    mapinfo.load_by_name(args.map_name)

    # Spatial keywords
    print("Loading spatial keywords...")
    with open(FILEPATHS["relation_keywords"]) as f:
        spatial_keywords = set(json.load(f))

    # Do this in a random order
    files = os.listdir(args.sg_dir)
    random.shuffle(files)

    try:
        accepted = []
        for filename in files:
            with open(os.path.join(args.sg_dir, filename)) as f:
                sg_dict = json.load(f)
            if sg_dict["map_name"] != args.map_name:
                continue

            print_sg_dict(sg_dict)
            accept = input_default("Accept? y/n", "n")
            if accept == "y":
                accepted.append(sg_dict)

            else:
                annotate = input_default("Annotate? y/n", "y")
                if annotate == "y":
                    sg_ann = do_annotation(sg_dict, mapinfo, spatial_keywords)
                    if sg_ann is not None:
                        accept = input_default("Accept? y/n", "y")
                        if accept == "y":
                            accepted.append(sg_ann)
                            print_sg_ann(sg_ann)
            print("You have accepted %d languages\n" % len(accepted))
    except EOFError:
        pass
    finally:
        print_error("Stopping...")
        if len(accepted) > 0:
            os.makedirs(args.outdir, exist_ok=True)
        for i, sg_dict in enumerate(accepted):
            filename = "sg-%s-%d.json" % (args.map_name, i)
            with open(os.path.join(args.outdir, filename), "w") as f:
                json.dump(sg_dict, f, indent=4, sort_keys=True)
                print_info("saved %s" % filename)
        print_success("DONE!")

if __name__ == "__main__":
    main()
