import pytest
import spacy
import sloop.observation
from sloop.osm.datasets import MapInfoDataset, FILEPATHS

def test_parse_to_splobservation():
    mapinfo = MapInfoDataset()
    mapinfo.load_by_name("austin")

    lang = "The red car is behind Lavaca Street."
    spacy_model = spacy.load("en_core_web_md")
    o = sloop.observation.parse(lang, "austin",
                                kwfile=FILEPATHS["symbol_to_synonyms"],
                                spacy_model=spacy_model,
                                verbose_level=1)
    assert o.relations[0] == ("RedHonda", "LavacaSt", "behind")
