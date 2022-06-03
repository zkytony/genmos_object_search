import os
import torch
import spacy
import json
import matplotlib.pyplot as plt
import sloop.observation
from sloop.osm.datasets import MapInfoDataset, FILEPATHS
from sloop.osm.datasets.utils import make_context_img
from sloop.osm.models.heuristics.test import plot_belief
from sloop.osm.models.heuristics.model import MixtureSLUModel
from sloop.osm.models.nn.plotting import plot_foref
import sloop.osm.models.heuristics.rules as R


def unittest():
    print("Loading synonyms...")
    with open(FILEPATHS["relation_keywords"]) as f:
        predicates = json.load(f)

    # Example on dorrance
    mapinfo = MapInfoDataset()
    map_name = "washington_dc"
    landmark = "SupportBuilding"
    mapinfo.load_by_name(map_name)
    print(mapinfo.center_of_mass(landmark))

    # Load a frame of reference prediction model
    path_to_foref_model =\
        os.path.join("../../models/iter2_ego-ctx-foref-angle:front:austin", "front_model.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(path_to_foref_model, map_location=device)

    print("Loading spacy model...")
    spacy_model = spacy.load("en_core_web_md")

    lang = "The red car is in front of Support Building."
    splang_obz = sloop.observation.parse(lang, map_name,
                                         kwfile=FILEPATHS["symbol_to_synonyms"],
                                         spacy_model=spacy_model,
                                         verbose_level=1)

    rules = {"near": R.NearRule(),
             "at": R.AtRule(),
             "beyond": R.BeyondRule(),
             "between": R.BetweenRule(),
             "east": R.DirectionRule("east"),
             "west": R.DirectionRule("west"),
             "north": R.DirectionRule("north"),
             "south": R.DirectionRule("south"),
             "front": R.ForefRule("front"),
             "behind": R.ForefRule("behind")}

    # spatial language observation model
    splang_om = MixtureSLUModel(rules, mapinfo,
                                foref_models={"front": model.predict_foref,
                                              "behind": model.predict_foref},
                                foref_kwargs={"device": device})
    result = splang_om.interpret(splang_obz)

    foref = result[1]["forefs"][0][2]
    map_arr = make_context_img(mapinfo, map_name, landmark,
                               dist_factor=2.0)
    plt.imshow(map_arr.transpose(), alpha=0.5)
    oo_belief = result[0]
    for obj in oo_belief:
        plot_belief(oo_belief[obj], ax=plt.gca())
    plot_foref(foref, plt.gca())
    plt.xlim(0, map_arr.shape[1])
    plt.ylim(0, map_arr.shape[0])
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    unittest()
