import pomdp_py
import sloop

class SpatialLanguageObservation(pomdp_py.Observation):
    """
    We define a spatial language observation as a set of
    (f, r, γ) tuples extracted from a given spatial language.
    """
    def __init__(self, splang_tuples):
        """
        Args:
            splang_tuples (tuple): a tuple of
                of tuples of the form (figure, spatial_relation, ground)
        """
        self.splang_tuples = splang_tuples

    def __hash__(self):
        return hash(self.splang_tuples)

    def __eq__(self, other):
        if isinstance(other, SpatialLanguageObservation):
            return self.splang_tuples == other.splang_tuples
        return False



    def _parse_query(self, query, map_name,
                     symbol_to_synonyms=None,
                     spatial_keywords=None,
                     spacy_model=None):
        """
        Given a `query`, which can either be:
        - a natural language sentence (str)
        - a sg_dict, i.e. spatial graph to dict (dict)
        - a tuple (ObjectSymbol, LandmarkSymbol, Keyword)
        Return sg_dict
        """
        if symbol_to_synonyms is None:
            print("Loading symbol to synonyms...")
            with open(FILEPATHS["symbol_to_synonyms"]) as f:
                symbol_to_synonyms = json.load(f)
        if spatial_keywords is None:
            print("Loading spatial keywords...")
            with open(FILEPATHS["relation_keywords"]) as f:
                spatial_keywords = json.load(f)
        if type(query) == str:
            if spacy_model is None:
                print("Loading spacy model...")
                spacy_model = spacy.load("en_core_web_md")

            sg = parse(query, map_name,
                       all_keywords=symbol_to_synonyms,
                       spacy_model=spacy_model)
            sg_dict = sg.to_dict()
        elif type(query) == dict:
            sg_dict = query
        elif type(query) == tuple:
            sg_dict = {"relations": [query]}
        else:
            raise ValueError("Unable to understand input query %s" % str(query))

        for i, rel in enumerate(sg_dict["relations"]):
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
            sg_dict["relations"][i] = (rel[0], rel[1], best_match)
        return sg_dict


def parse(query, map_name, **kwargs):
    """
    A wrapper around the 'parse' function in sloop.parsing.parser,
    which parses a given spatial language using spacy. This function
    will convert that output to a SpatialLanguageObservation.

    Args:
        query: can either be:
               - a natural language sentence (str)
               - a sg_dict, i.e. spatial graph to dict (dict)
               - a tuple (ObjectSymbol, LandmarkSymbol, Keyword)

        See the sloop.parsing.parser.parse function for documentation of
        **kwargs.

    Returns:
        SpatialLanguageObservation

    Example use (with OSM maps):
    >>> import spacy
    >>> from sloop.observation import parse
    >>> from sloop.osm.datasets import MapInfoDataset, FILEPATHS
    >>>
    >>> mapinfo = MapInfoDataset()
    >>> mapinfo.load_by_name("austin")
    >>>
    >>> lang = "The red car is behind Lavaca Street."
    >>> spacy_model = spacy.load("en_core_web_md")
    >>> parse(lang, "austin",
    >>>       kwfile=FILEPATHS["symbol_to_synonyms"],
    >>>       spacy_model=spacy_model,
    >>>       verbose_level=1)
    """
    if type(query) == str:
        if spacy_model is None:
            print("Loading spacy model...")
            spacy_model = spacy.load("en_core_web_md")

        sg = parse(query, map_name,
                   all_keywords=symbol_to_synonyms,
                   spacy_model=spacy_model)
        sg_dict = sg.to_dict()
    elif type(query) == dict:
        sg_dict = query
    elif type(query) == tuple:
        sg_dict = {"relations": [query]}
    else:
        raise ValueError("Unable to understand input query %s" % str(query))

    for i, rel in enumerate(sg_dict["relations"]):
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
        sg_dict["relations"][i] = (rel[0], rel[1], best_match)
    return SpatialLanguageObservation(tuple(sorted(sg_dict['relations'])))
