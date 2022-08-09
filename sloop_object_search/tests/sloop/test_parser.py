import pytest
import spacy
from sloop.parsing import parser
from sloop.osm.datasets import MapInfoDataset, FILEPATHS

def test_parser_basic():
    mapinfo = MapInfoDataset()
    mapinfo.load_by_name("austin")

    lang = "The red car is behind Lavaca Street."
    spacy_model = spacy.load("en_core_web_md")
    parser.parse(lang, "austin",
                 kwfile=FILEPATHS["symbol_to_synonyms"],
                 spacy_model=spacy_model,
                 verbose_level=1)
