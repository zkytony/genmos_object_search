import csv
import json
import spacy
import copy
import os
import argparse
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.parsing.parser import *

class DataSample:
    def __init__(self, hint, sg, map_link, map_name):
        self.hint = hint
        self.sg = sg
        self.map_link = map_link
        self.map_name = map_name

    def to_dict(self):
        return {"hint": self.hint,
                "sg": self.sg.to_dict(),
                "map_link": self.map_link,
                "map_name": self.map_name}

class AMTCSVLoader:
    def __init__(self, spacy_model_name="en_core_web_sm"):
        if spacy_model_name is not None:
            print("Loading spacy model...")
            self.spacy_model = spacy.load(spacy_model_name)

            print("Loading symbol to synonyms...")
            with open(FILEPATHS["symbol_to_synonyms"]) as f:
                self.symbol_to_synonyms = json.load(f)

    def get_result(self, reader, row_start, row_end, keyword=None):
        samples = []
        for i, row in enumerate(reader):
            if i < row_start or i >= row_end:
                continue
            print("Parsing row %d" % i)
            hint = row["hint"]
            if keyword is not None and keyword not in hint:
                print("row %d skipped due to: Keyword %s not present" % (i, keyword))
                continue
            try:
                sg = parse(hint, row["map_name"],
                           all_keywords=self.symbol_to_synonyms,
                           spacy_model=self.spacy_model)
            except Exception as ex:
                print("row %d skipped due to: %s" % (i, str(ex)))
                continue
            data_sample = DataSample(hint, sg, row["map_link"], row["map_name"])
            samples.append(data_sample.to_dict())
        print("Got %d samples" % len(samples))
        return samples

    def parse_result(self, reader, path_to_result, output_dirpath, name="annot"):
        """Write sg files"""
        # First read all of the csv file and create a dictionary
        data = {}
        duplicated_hints = set()
        for i, row in enumerate(reader):
            # If the hint is already stored, then there is a duplication.
            # Mark it, and skip this sample later.
            if row["hint"] in data:
                duplicated_hints.add(row["hint"])
            data[row["hint"]] = row

        with open(path_to_result) as f:
            saved_result = json.load(f)

        for i, sample in enumerate(saved_result["samples"]):
            output_sample = {}
            hint = sample["hint"]
            if hint in duplicated_hints:
                print("Hint: \"%s\" has duplications. Skipped." % hint)
                continue

            if hint in data:
                output_sample = copy.deepcopy(data[hint])
                output_sample["entities"] = sample["sg"]["entities"]
                output_sample["relations"] = sample["sg"]["relations"]
                output_sample["frame_of_refs_pixels"] =  sample["sg"]["frame_of_refs_pixels"]
                output_sample["frame_of_refs_grid"] =  sample["sg"]["frame_of_refs_grid"]
            else:
                raise ValueError("%s not in csv file. Something wrong." % hint)
            save_path = os.path.join(output_dirpath, "sg-%s-%d.json" % (name, i))
            with open(save_path, "w") as f:
                json.dump(output_sample, f, indent=4, sort_keys=True)
                print("Saved sample %d to %s" % (i, save_path))



def unittest():
    with open(FILEPATHS["amt_fau_dor_csv"]) as f:
        reader = csv.DictReader(f, delimiter=",")
        amt = AMTCSVLoader(reader)
        samples = amt.get_result(0, 5)
        print(json.dumps(samples))

if __name__ == "__main__":
    # unittest()
    parser = argparse.ArgumentParser(description='Process Saved Annotation Results.')
    parser.add_argument('path_to_csv', type=str,
                        help='Path to the amt csv file')
    parser.add_argument('path_to_downloaded_json', type=str,
                        help='Path to the directory that contains the .json files you downloaded from the annotator')
    parser.add_argument('output_dir', type=str,
                        help='Directory where you want to output the sg json files (One sg json file per sample).')
    args = parser.parse_args()

    with open(args.path_to_csv) as f:
        reader = csv.DictReader(f, delimiter=',')

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        amt = AMTCSVLoader(spacy_model_name=None)
        for filename in os.listdir(args.path_to_downloaded_json):
            if filename.endswith("json"):
                print("Processing %s" % filename)
                path_to_file = os.path.join(args.path_to_downloaded_json, filename)
                name = os.path.splitext(filename)[0]
                amt.parse_result(reader, path_to_file, args.output_dir, name=name)
                f.seek(0)
