import pomdp_py
import spacy
import sloop.parsing.parser
from .constants import SPATIAL_KEYWORDS

class SpatialLanguageObservation(pomdp_py.Observation):
    """
    We define a spatial language observation as a set of
    (f, r, γ) tuples extracted from a given spatial language.

    Note that in the implementation the tuple is actually
    (f, γ, r), because the dataset had been collected that way.
    """
    def __init__(self, map_name, relations):
        """
        Args:
            map_name (str): name of the map where everything lives in
            relations (tuple): a tuple of
                of tuples of the form (ObjectSymbol, LandmarkSymbol, Keyword)
                which is essentially (figure, ground, spatial_relation)
        """
        self.relations = relations
        self.map_name = map_name

    def __hash__(self):
        return hash(self.relations)

    def __eq__(self, other):
        if isinstance(other, SpatialLanguageObservation):
            return self.relations == other.relations
        return False

    def __iter__(self):
        return iter(self.relations)

    def __str__(self):
        return str(self.relations)

    def __repr__(self):
        return f"SpatialLanguageObservation({str(self)})"


def parse(query, map_name, spatial_keywords=SPATIAL_KEYWORDS, **kwargs):
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
        spacy_model = kwargs.get("spacy_model")
        if spacy_model is None:
            print("Loading spacy model...")
            spacy_model = spacy.load("en_core_web_lg")
            kwargs["spacy_model"] = spacy_model
        sg = sloop.parsing.parser.parse(query, map_name, **kwargs)
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
            best_match = sloop.parsing.parser.match_spatial_keyword(
                rel_phrase,
                spatial_keywords,
                similarity_thres=0.99,
                spacy_model=spacy_model,
                match_multiple=False)
        else:
            best_match = rel_phrase  # rel phrase is just a word
        sg_dict["relations"][i] = (rel[0], rel[1], best_match)
    return SpatialLanguageObservation(map_name, tuple(sorted(sg_dict['relations'])))
