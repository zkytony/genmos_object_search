# Calculates the precision and recall of our parsing pipeline
# based on annotated vs automatically parsed language

import os
import json
import spacy
from sloop_object_search.parsing.parser import parse, match_spatial_keyword
from sloop_object_search.data.constants import FILEPATHS
from sloop_object_search.models.heuristics.rules import BASIC_RULES


def main():
    print("Loading spacy model...")
    spacy_model = spacy.load("en_core_web_md")

    # Load language symbol files
    print("Loading symbol to synonyms...")
    with open(FILEPATHS["symbol_to_synonyms"]) as f:
        symbol_to_synonyms = json.load(f)

    print("Loading spatial keywords...")
    with open(FILEPATHS["relation_keywords"]) as f:
        spatial_keywords = json.load(f)

    # Load the annotated languages into a map from language to sgdicts
    dataset_path = "../../datasets/SL-OSM-Dataset"
    annot_sgs = {}
    for map_name in {"austin", "cleveland", "denver", "honolulu", "washington_dc"}:
        dirpath = os.path.join(dataset_path, "sg_parsed_annotated", map_name)
        for filename in os.listdir(dirpath):
            with open(os.path.join(dirpath, filename)) as f:
                sg = json.load(f)
        annot_sgs[sg["lang_original"]] = sg

    # Parse these languages, record the relations
    lang2case = {}
    for lang in annot_sgs:
        map_name = annot_sgs[lang]["map_name"]

        sg_auto = parse(lang, map_name,
                    all_keywords=symbol_to_synonyms,
                    spacy_model=spacy_model).to_dict()
        for i, rel in enumerate(sg_auto["relations"]):
            rel_phrase = rel[2]
            if rel_phrase is not None and\
               len(rel_phrase.split(" ")) > 1:
                best_match = match_spatial_keyword(
                    rel_phrase,
                    spatial_keywords,
                    similarity_thres=0.99,
                    spacy_model=spacy_model,
                    match_multiple=False)
            else:
                best_match = rel_phrase  # rel phrase is just a word
            sg_auto["relations"][i] = (rel[0], rel[1], best_match)

        lang2case[lang] = {"auto":set(), "annot":set()}
        for rel in sg_auto["relations"]:
            if rel[2] not in BASIC_RULES:
                continue
            lang2case[lang]["auto"].add(rel)
        sg_annot = annot_sgs[lang]
        for rel in sg_annot["relations"]:
            if rel[2] not in BASIC_RULES:
                continue
            lang2case[lang]["annot"].add(tuple(rel))

    # Calculate precision/recall of landmarks
    true_pos_lmk = 0
    false_neg_lmk = 0
    false_pos_lmk = 0
    for lang in lang2case:
        landmarks_annotated = set()
        landmarks_parsed = set()
        for rel in lang2case[lang]["annot"]:
            landmark = rel[1]
            landmarks_annotated.add(landmark)
        for rel in lang2case[lang]["auto"]:
            landmark = rel[1]
            landmarks_parsed.add(landmark)

        for lmk in landmarks_annotated:
            if lmk in landmarks_parsed:
                true_pos_lmk += 1
            else:
                false_neg_lmk += 1

        for lmk in landmarks_parsed:
            if lmk not in landmarks_annotated:
                false_pos_lmk += 1

    print("---- Landmark Recognition ----")
    print("precision: %.3f" % (true_pos_lmk / (true_pos_lmk + false_pos_lmk)))
    print("recall: %.3f" % (true_pos_lmk / (true_pos_lmk + false_neg_lmk)))

    # Calculate precision/recall of (f,r,g) tuples
    true_pos_rel = 0
    false_neg_rel = 0
    false_pos_rel = 0
    for lang in lang2case:
        tuples_annotated = set()
        tuples_parsed = set()
        for rel in lang2case[lang]["annot"]:
            tuples_annotated.add(rel)
        for rel in lang2case[lang]["auto"]:
            tuples_parsed.add(rel)

        for rel in tuples_annotated:
            if rel in tuples_parsed:
                true_pos_rel += 1
            else:
                false_neg_rel += 1

        for rel in tuples_parsed:
            if rel not in tuples_annotated:
                false_pos_rel += 1

    print("---- Relation Tuple Recognition ----")
    print("precision: %.4f" % (true_pos_rel / (true_pos_rel + false_pos_rel)))
    print("recall: %.4f" % (true_pos_rel / (true_pos_rel + false_neg_rel)))


if __name__ == "__main__":
    main()
