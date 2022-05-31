# Parses all languages into json files.True

from .parser import *
import csv
import argparse
import os
import spacy

def main():
    parser = argparse.ArgumentParser(description="Parse a language")
    parser.add_argument("-datafile", type=str, action='store', default="../data/processed_data/amt_fau_dor_data.csv",
                        help="Path to .csv data file.")
    parser.add_argument("-kwfile", type=str, action='store', default="../data/language/fau_dor_keywords.json",
                        help="Path to JSON a file with definition of keywords"\
                        "(e.g. landmarks, objects). Normally at spatial_lang/data/language/")
    parser.add_argument("-output_dirpath", action='store', type=str, default="../data/relations/graphs_data2/output",
                        help="Path to output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dirpath):
        print("Creating output directory...")
        os.makedirs(args.output_dirpath)

    print("Loading spacy model...")
    spacy_model = spacy.load("en_core_web_md")

    print("Reading keywords file...")
    with open(args.kwfile) as f:
        all_keywords = json.load(f)
    print("Reading data file...")
    with open(args.datafile) as f:
        reader = csv.DictReader(f, delimiter=",")
        for i, row in enumerate(reader):
            row["index"] = i
            row["lang_original"] = row["hint"]
            del row["hint"]
            print("Parsing language at row %d" % i)
            try:
                sg = parse(row["lang_original"], row["map_name"],
                           args.kwfile,
                           spacy_model=spacy_model)
                filename = "sg-%d.json" % i
                sg.to_file(os.path.join(args.output_dirpath, filename),
                           **row)
            except Exception as ex:
                print("Failed to parse row %d: (err: %s. %s)\n  \"%s\""\
                      % (i, type(ex),
                         str(ex), row["lang_original"]))

if __name__ == "__main__":
    main()
