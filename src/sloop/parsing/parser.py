# Uses spacy.
import csv
import argparse
import json
import spacy
from pprint import pprint
import os
import re
import time
from . import spacy_utils as sputils
from .graph.spatial import *
from .deptree import *

def parse(lang, map_name,
          kwfile=None,
          all_keywords={},
          spacy_model=None,
          verbose_level=0):
    """Parses the language knowing that there are keywords
    given by the `kwfile` and the keywords dictionary.

    spacy_model is the result of `spacy.load`. Supply this
    to avoid repeatedly loading the model.

    Args:
        lang (str): A string of the spatial language we want to parse
        kwfile (str): Path to the keywords file
        all_keywords (dict): An alternative way to pass in keywords info
        spacy_model (spaCy model): The result of loading a spacy model (e.g. "en")
        verbose_level (int): For debugging purpose
    Returns:
        SpatialGraph: a spatial graph representation of the spatial language
    """
    # Read keywords file
    if kwfile is not None:
        with open(kwfile) as f:
            all_keywords = json.load(f)
    else:
        assert len(all_keywords) > 0, "Requires either kwfile or keywords"

    # Parse the language with spacy and get all the noun phrases.
    # Filter based on distance to keywords
    if spacy_model is None:
        spacy_model = spacy.load("en_core_web_sm")
    doc = spacy_model(lang)

    # This is for optimization; preloading the Doc objects for keywords.
    all_keywords = _prepare_keyword_docs(all_keywords, spacy_model, map_name)

    if verbose_level > 1:
        print("Nown phrases:")
        for noun_span in sputils.noun_chunks(doc):
            print(noun_span)
        print("----")

    # We want to know if we need to break down the given sentence (because
    # of multiple subjects). Because we also substitute symbols, this may
    # change the dependency parsing behavior. So we break down both
    # the original version and the substituted version. And we proceed
    # with the break-down with the most number of sentences
    subdocs_original = _break_down(doc)  # sub here means "a smaller part"
    # Substitute for keywords
    lang_new, obj_matches, landmark_matches, rel_matches =\
        _substitute_keywords(doc, all_keywords, spacy_model, map_name,
                             verbose_level=verbose_level)
    # Get doc for the new language
    doc_new = spacy_model(lang_new)
    subdocs_new = _break_down(doc_new)

    using_new_doc = len(subdocs_new) > len(subdocs_original)
    if using_new_doc:
        subdocs = subdocs_new
    else:
        subdocs = subdocs_original

    # Parse each subdoc (a sentence) separately.
    sg_list= []
    subdoc_substitute_exceptions = 0
    for i, subdoc in enumerate(subdocs):
        if verbose_level > 0:
            print("\nSubdoc %d: %s" % (i, subdoc.text))
            print("Using new doc? %s" % str(using_new_doc))
        if not using_new_doc:
            # we need to do substitution since these subdocs are not symbolized
            try:
                sublang_new, obj_matches, landmark_matches, rel_matches =\
                    _substitute_keywords(subdoc, all_keywords, spacy_model, map_name,
                                         verbose_level=verbose_level)
                subdoc = spacy_model(sublang_new)
            except ValueError:
                subdoc_substitute_exceptions += 1
            if verbose_level > 0:
                print(obj_matches)
                print(landmark_matches)
        for sent in subdoc.sents:
            deptree = _build_deptree(sent.as_doc(), verbose_level=verbose_level)
            paths = _find_paths(deptree, obj_matches, landmark_matches,
                                verbose_level=verbose_level)
            sg = _build_spatial_graph(paths, subdoc.text, obj_matches, landmark_matches,
                                      verbose_level=verbose_level)
            sg_list.append(sg)
            if verbose_level > 0:
                print("Spatial Graph for Subdoc %d" % i)
                print(sg.to_dict())
    if not(len(subdocs) - subdoc_substitute_exceptions > 0):
        # Too many substitution errors - not a single subdoc (i.e. sentence) is
        # valid for substitution.
        raise ValueError("The language \"%s\" does not contain a single subdoc (i.e. sentence)"
                         "valid for substitution." % lang)
    return SpatialGraph.join(sg_list)

def _prepare_keyword_docs(all_keywords, spacy_model, map_name):
    """For optimization, create the spacy.Doc objects for
    all keywords."""
    res = {}
    for catg in all_keywords:
        res[catg] = {}
        if catg.lower() in {"objects", "landmarks_%s" % map_name,
                            "spatial_relations", "swaps"}:
            for symbol in all_keywords[catg]:
                res[catg][symbol] = []
                for synonym in all_keywords[catg][symbol]:
                    kwdoc = spacy_model(synonym)
                    res[catg][symbol].append(kwdoc)
        else:
            res[catg] = all_keywords[catg]
    return res

def _break_down(doc):
    """Break down the given language Doc into multiple sentences used for
    parsing individually. Relies on the Spacy dependency parsing
    capability.

    We basically break down sentences by nsubj (noun subjects)."""
    if len(list(doc.sents)) > 1:
        docs = [sent.as_doc() for sent in doc.sents]
        splitted_docs = []
        for doc in docs:
            splitted_docs.extend(_break_down(doc))
        return splitted_docs

    nsubjs = []
    for token in doc:
        if token.dep_.lower().startswith("nsubj"):
            # We obtain the tokens in the subtree of this token
            nsubj_tokens = list(token.subtree)
            nsubjs.append(doc[nsubj_tokens[0].i : nsubj_tokens[-1].i])
    if len(nsubjs) == 1:
        return [doc]
    # More than one subject. Then split the doc by the subject tokens
    splits = []
    start = 0
    end = -1
    for nsubj_span in nsubjs[1:]:
        # Check the word prior to the first token of this span. Skip "and" or ","
        end = nsubj_span.start
        if nsubj_span.start-1 > 0:
            if doc[nsubj_span.start-1].text.lower() in {"and", ","}:
                end = end - 1
        splits.append((start, end))
        start = nsubj_span.start
    # Add the last split
    splits.append((start, len(doc)))

    # Split the doc by splits
    splitted_docs = []
    for start, end in splits:
        splitted_docs.append(doc[start:end].as_doc())
    return splitted_docs

def _simi_thres(catg, all_keywords, default=0.8):
    """Return a similarity threshold, either from the specification in
    all_keywords (dict) or using the default."""
    if "_thresholds_" not in all_keywords:
        return default
    elif catg not in all_keywords["_thresholds_"]:
        return default
    else:
        return all_keywords["_thresholds_"][catg]

def _substitute_keywords(doc, all_keywords, spacy_model, map_name, verbose_level=0):
    """Given a spacy Doc, and specification of keywords (all_keywords),
    return a string that is the language in the given doc with synonyms of keywords
    replaced by a canonical symbol."""
    lang_original = doc.text

    # all_keywords may have a field "swap" which is used for
    # matching and swapping phrases. For example, swapping
    # 'two cars" into "the green toyota and the blue honda'
    if "swaps" in all_keywords:
        swap_matches = []
        for noun_span in sputils.noun_chunks(doc):
            swap_symbol = _match(noun_span, all_keywords["swaps"], spacy_model,
                                     similarity_thres=_simi_thres("swaps", all_keywords))
            if swap_symbol is not None:
                swap_matches.append((swap_symbol, noun_span))
        # actually do the swap, and get a new doc object
        lang_new = doc.text
        for swap_symbol, noun_span in swap_matches:
            lang_new = lang_new.replace(noun_span.text, swap_symbol)
        doc = spacy_model(lang_new)

        if verbose_level > 0:
            print("Swapping:")
            print("  before swap: \"%s\"" % lang_original)
            print("  after swap: \"%s\"" % doc.text)

    # Match objects, landmarks, relations
    matched_tokens = set({})

    start = time.time()

    # match objects
    obj_matches = []
    for noun_span in sputils.noun_chunks(doc):
        if noun_span[0].text.lower() == "the":
            noun_span = noun_span[1:]  # forget about "the"
        obj_symbol = _match(noun_span, all_keywords["objects"], spacy_model,
                           similarity_thres=_simi_thres("objects", all_keywords))
        if obj_symbol is not None:
            obj_matches.append((obj_symbol, noun_span))
            matched_tokens.update(list(noun_span))
    if len(obj_matches) == 0:
        raise ValueError("The given language: \"%s\" does not appear to have a noun phrase "\
                         "that matches any keyword phrases for any target object"\
                         % (doc.text))
    if verbose_level > 1:
        print("[TIME] Object matching took %.3fs" % (time.time() - start))
        start = time.time()

    # match landmarks
    landmark_matches = []
    for noun_span in sputils.noun_chunks(doc):
        landmark_symbol = _match(noun_span, all_keywords["landmarks_%s" % map_name],
                                 spacy_model, similarity_thres=_simi_thres("landmarks", all_keywords))
        if landmark_symbol is not None:
            landmark_matches.append((landmark_symbol, noun_span))
            matched_tokens.update(list(noun_span))
    if verbose_level > 1:
        print("[TIME] Landmark matching took %.3fs" % (time.time() - start))
        start = time.time()

    # match relational keywords.
    rel_matches = []
    for token in doc:
        if token not in matched_tokens:
            rel_symbol = _match(token, all_keywords["spatial_relations"], spacy_model,
                               similarity_thres=_simi_thres("spatial_relations", all_keywords))
            if rel_symbol is not None:
                # try to hook up this word with the word after it, if the word
                # after it is a POBJ (otherwise this relational symbol will not show up
                # in the path). Cases that can't be fixed, we'll have to live with it.
                if token.i + 1 < len(doc):
                    next_token = doc[token.i + 1]
                    rel_symbol = "with %s" % rel_symbol
                    if next_token.text.lower() != "of" or next_token.dep_.lower() == "pobj":
                        # insert an "of" to influence dependency parsing and "on" before it
                        rel_symbol = "%s of" % rel_symbol
                    rel_matches.append((rel_symbol, token))
                    matched_tokens.add(token)
    if verbose_level > 1:
        print("[TIME] Relation matching took %.3fs" % (time.time() - start))
        start = time.time()

    if verbose_level > 1:
        print("Original: \"%s\"" % lang_original)
        print("Parsing: \"%s\"" % doc.text)
        print("Matched objects:")
        print(obj_matches)
        print("Matched Landmarks:")
        pprint(landmark_matches)
        print("Matched Relational Keywords:")
        pprint(rel_matches)

    # Replace noun phrases in the original sentence by single symbol.
    lang_new = doc.text
    for obj_symbol, noun_span in obj_matches:
        lang_new = lang_new.replace(noun_span.text, obj_symbol)
    for landmark_symbol, noun_span in landmark_matches:
        lang_new = lang_new.replace(noun_span.text, landmark_symbol)
    for rel_symbol, noun_span in rel_matches:
        lang_new = lang_new.replace(" %s " % (noun_span.text),
                                    " %s " % rel_symbol)
    if verbose_level > 0:
        print("Substituting symbols back to original sentence:\n  \"%s\"" % lang_new)
    return lang_new, obj_matches, landmark_matches, rel_matches


def _build_deptree(doc, verbose_level=0):
    # build dependency tree with spaCy!
    deptree = DependencyTree.build(doc, None, method="spacy")
    if verbose_level > 0:
        print("Printing dependency tree")
        deptree.pprint()
    if verbose_level > 1:
        print("[TIME] Substitution and dependency tree building took %.3fs" % (time.time() - start))
        start = time.time()
    return deptree

def _find_paths(deptree, obj_matches, landmark_matches, verbose_level=0):
    """Find paths between the entities. Returns a list of paths"""
    # Get the paths between from every mention of the object to every landmark.
    paths = []
    for obj_symbol, _ in obj_matches:
        nobj_res = deptree.lookup("content", obj_symbol, err_if_unfound=False)
        for node_obj in nobj_res:
            for landmark_symbol, _ in landmark_matches:
                nlmk_res = deptree.lookup("content", landmark_symbol, err_if_unfound=False)
                for node_landmark in nlmk_res:
                    path = deptree.path_between(node_obj, node_landmark)
                    paths.append((obj_symbol, landmark_symbol, path))
                    if verbose_level > 2:
                        print("Path between\n  %s\nand\n  %s:" % (node_obj, node_landmark))
                        for i, node in enumerate(path):
                            if i != 0 and i != len(path)-1:
                                print("%d. *%s" % (i+1, node))
                            else:
                                print("%d. %s" % (i+1, node))
    if verbose_level > 1:
        print("[TIME] Path finding took %.3fs" % (time.time() - start))
        start = time.time()
    return paths

def _build_spatial_graph(paths, lang_new, obj_matches, landmark_matches, verbose_level=0):
    """
    Given paths (output of _find_paths), the language (lang_new, it is the
    substituted version of the original language), and the matches,
    return a SpatialGraph
    """
    # Get the words connecting the object and landmark - that will be the relations
    # Then create spatial graphs
    entities = set({obj_symbol for obj_symbol, _ in obj_matches})\
        | set({landmark_symbol for landmark_symbol, _ in landmark_matches})
    dct = {"entities": entities,
           "relations":[],
           "lang": lang_new}
    for obj_symbol, landmark_symbol, path in paths:
        spatial_relation_label = ""
        for i, node in enumerate(path[1:-1]):
            # There may be many heuristic-based things to be done here
            # to make the spatial relation more reasonable to deal with.
            if node.content in dct["entities"]:  # skip the object symbols
                continue
            spatial_relation_label += node.content
            if i < len(path[1:-1]) - 1:
                spatial_relation_label += " "
        if len(spatial_relation_label) > 0:
            dct["relations"].append((obj_symbol, landmark_symbol, spatial_relation_label))
    if verbose_level > 1:
        print("[TIME] Creating SpatialGraph %.3fs" % (time.time() - start))
        start = time.time()
    return SpatialGraph.from_dict(dct)


def _match(noun_token, keywords,
          spacy_model, similarity_thres=0.8):
    """
    keywords: dictionary, "<canonical_name>" -> ["synonyms"...]
    """
    similarity_scores = {}  # obj -> similarity score
    matching_synonyms = {}  # obj -> number of synonyms have high similarity score
    for item in keywords:
        similarity_scores[item] = 0
        matching_synonyms[item] = 0

        for synonym in keywords[item]:
            if type(synonym) == str:
                synonym = spacy_model(synonym)
            else:
                assert isinstance(synonym, spacy.tokens.Doc),\
                    "Expecting the synonym %s for keyword matching to be a spacy Doc."\
                    % str(synonym)

            similarity = noun_token.similarity(synonym)
            if similarity > similarity_thres:
                similarity_scores[item] = max(similarity_scores[item], similarity)
                matching_synonyms[item] += 1

    # Among the items with a score, find one with highest.
    # If no item has any score, then no match.
    best_item, best_score = None, 0.0
    for item in matching_synonyms:
        if matching_synonyms[item] > 0:
            if similarity_scores[item] > best_score:
                best_score = similarity_scores[item]
                best_item = item
    return best_item

def match_spatial_keyword(rel_phrase,
                          spatial_keywords, similarity_thres=0.98, spacy_model=None,
                          match_multiple=False, err_on_multiple=False,
                          filler_words={"is", "of", "to", "are", "the", "with"},
                          low_priority_words={"on", "at", "in", "near", "right"}):
    """
    This function assumes the `rel_phrase` is a relational phrase
    computed as result of `parse`. It returns a best matched keyword,
    or multiple matched keywords given the list of keywords in `spatial_keywords`.
    """
    def _fix_whitespace(sent):
        """Make sure the words in the sentence `sent` are separated by
        one single white space."""
        return re.sub("[\s|\n]+", " ", sent)

    if spacy_model is None:
        spacy_model = spacy.load("en_core_web_sm")
    keyword_docs = {kw: spacy_model(kw) for kw in spatial_keywords}
    # remove unnecessary words
    rel_phrase = _fix_whitespace(rel_phrase.replace(" of ", " "))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right near", "near"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right next to", "next"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right behind", "behind"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right above", "above"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right below", "below"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right at", "at"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right in", "in"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right by", "by"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right on", "on"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right between", "between"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("with ", " "))
    rel_phrase = _fix_whitespace(rel_phrase.replace("on on", "on"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("is is", "on"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("is on", "on"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("is by", "by"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("parked", ""))
    rel_phrase = _fix_whitespace(rel_phrase.replace("alongside", "along"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("located", ""))
    rel_phrase = _fix_whitespace(rel_phrase.replace("in front", "front"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("in back", "back"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right next", "next"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right below", "below"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right above", "above"))
    rel_phrase = _fix_whitespace(rel_phrase.replace("right at", "at"))
    rel_phrase_doc = spacy_model(rel_phrase)

    best_kw = None
    best_similarity = -1
    matched_kw = set({})
    for word_doc in rel_phrase_doc:
        if not word_doc.text.isspace()\
           and word_doc and word_doc.norm:
            if word_doc.text not in filler_words:
                for kw in keyword_docs:
                    simi = keyword_docs[kw].similarity(word_doc)
                    if simi > similarity_thres:
                        matched_kw.add(kw)
                        if simi > best_similarity:
                            best_similarity = simi
                            best_kw = kw
    # Prefer others over ones that have lower priority
    if best_kw in low_priority_words and len(matched_kw) > 1:
        for kw in matched_kw:
            if kw not in low_priority_words:
                best_kw = kw
                break
    if match_multiple:
        return best_kw, matched_kw
    else:
        if err_on_multiple and len(matched_kw) > 1:
            for mkw in matched_kw:
                if mkw not in low_priority_words and (mkw + " ") in rel_phrase and mkw != best_kw:
                    raise ValueError("\"%s\" contains multiple keywords" % rel_phrase)
        return best_kw


def main():
    parser = argparse.ArgumentParser(description="Parse a language")
    parser.add_argument("lang", type=str,
                        help="The sentence you want to parse")
    parser.add_argument("map_name", type=str,
                        help="map name (e.g. cleveland)")
    parser.add_argument("kwfile", type=str,
                        help="Path to JSON a file with definition of keywords"\
                        "(e.g. landmarks, objects). Typically the symbol_to_synonyms.json file.")
    args = parser.parse_args()
    print("Loading spacy model...")
    spacy_model = spacy.load("en_core_web_md")
    print("Parsing language")
    sg = parse(args.lang, args.map_name, args.kwfile,
               spacy_model=spacy_model, verbose_level=1)
    pprint(sg.to_dict())


if __name__ == "__main__":
    main()
