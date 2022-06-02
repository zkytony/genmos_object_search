from sloop.parsing.parser import parse
from sloop.parsing.parser import parse, match_spatial_keyword
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.models.heuristics.rules import *
from sloop.utils import smart_json_hook
from sloop.datasets.utils import make_context_img
from sloop.datasets.dataloader import *
from sloop.models.nn.metrics import *
import json
import spacy
import os, sys
import torch
import pomdp_py
from pprint import pprint

class InterpretModel:

    def interpret(self, query, map_name, mapinfo,
                  symbol_to_synonyms=None,
                  spatial_keywords=None,
                  **kwargs):
        """
        Given a `query`, which can either be:
        - a natural language sentence (str)
        - a sg_dict, i.e. spatial graph to dict (dict)
        - a tuple (ObjectSymbol, LandmarkSymbol, Keyword)
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map."""
        raise NotImplementedError

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


############# Rule based model ###########
class RuleBasedModel(InterpretModel):
    # Simple product SLU model

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""

    def __init__(self, rules):
        """
        `rules` is a mapping from keyword to a Rule
        """
        self._rules = rules

    def interpret(self, query, map_name, mapinfo,
                  symbol_to_synonyms=None,
                  spatial_keywords=None,
                  foref_models={}, foref_kwargs={},
                  spacy_model=None):
        """
        Given a `query`, which can either be:
        - a natural language sentence (str)
        - a sg_dict, i.e. spatial graph to dict (dict)
        - a tuple (ObjectSymbol, LandmarkSymbol, Keyword)
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map."""
        # The parse query here also matches keyword
        sg_dict = self._parse_query(query, map_name,
                                    symbol_to_synonyms=symbol_to_synonyms,
                                    spatial_keywords=spatial_keywords,
                                    spacy_model=spacy_model)

        oo_belief = {}  # mapping from object symbol to belief.
        landmark_footprints = {}
        rules_used = []
        forefs_list = []  # list of (landmark, keyword, frame of reference)
        for rel in sg_dict["relations"]:
            object_symbol, landmark_symbol, best_match = rel
            landmark_locs = mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs

            for keyword in [best_match]:  # TODO: REFACTOR
                if keyword not in self._rules:
                    if keyword == "front":
                        import pdb; pdb.set_trace()
                    print("Ignoring keyword '%s'. No rule provided." % keyword)
                    continue
                if isinstance(self._rules[keyword], ForefRule)\
                   and keyword not in foref_models:
                    print("Ignoring keyword '%s'. ForefRule, but no model provided."
                          % keyword)
                    continue
                print("Accepted keyword '%s'." % keyword)
                # Compute a belief distribution based on landmark locations
                # belief should be a mapping from (x,y) to a probability of object exist.
                kwargs = {}
                if isinstance(self._rules[keyword], ForefRule):
                    # obtain frame of reference
                    foref_model = foref_models[keyword]
                    foref = foref_model(keyword, landmark_symbol, map_name,
                                        mapinfo, **foref_kwargs)
                    if foref is None:
                        continue
                    kwargs["foref"] = foref
                    forefs_list.append((landmark_symbol, keyword, foref))
                belief = self._rules[keyword].compute(landmark_locs,
                                                      mapinfo.map_dims(map_name),
                                                      **kwargs)
                if object_symbol in oo_belief:
                    oo_belief[object_symbol] =\
                        combine_beliefs(oo_belief[object_symbol], belief, method="product")
                else:
                    oo_belief[object_symbol] = belief
                rules_used.append(keyword)

        # Cutoff tails
        for object_symbol in oo_belief:
            belief = oo_belief[object_symbol]
            oo_belief[object_symbol] = cutoff(belief, mapinfo.map_dims(map_name), landmark_footprints)
        return oo_belief, {"landmark_footprints": landmark_footprints,
                           "rules_used": rules_used,
                           "sg_dict": sg_dict,
                           "forefs": forefs_list}

############# Multi(level/interpretation) model ###########
class MixtureSLUModel(InterpretModel):

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""

    def __init__(self, rules, mixtures=[0,1,2,3], weights=[0.6, 0.25, 0.1, 0.05]):
        """
        `rules` is a mapping from keyword to a Rule
        mixtures (str or list): String ('all') or a list of
            numbers indicating mixtures to use. See number definitions:
            * mixture 0: joint everything
            * mixture 1: joint everything except for Foref relations
            * mixture 2: at rule for all landmarks
            * mixture 3: near rule for all landmarks
        weights (list of floats): List of float weights to combine the mixtures.
            The weights should sum up to 1.
        """
        self._rules = rules
        self.mixtures = mixtures
        self.weights = weights

    def interpret(self, query, map_name, mapinfo,
                  symbol_to_synonyms=None,
                  spatial_keywords=None,
                  foref_models={}, foref_kwargs={},
                  spacy_model=None):
        """
        Given a `query`, which can either be:
        - a natural language sentence (str)
        - a sg_dict, i.e. spatial graph to dict (dict)
        - a tuple (ObjectSymbol, LandmarkSymbol, Keyword)
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map."""
        # The parse query here also matches keyword
        sg_dict = self._parse_query(query, map_name,
                                    symbol_to_synonyms=symbol_to_synonyms,
                                    spatial_keywords=spatial_keywords,
                                    spacy_model=spacy_model)


        landmark_footprints = {}
        rules_used = []
        forefs_list = []
        oo_rels = {}  # mapping from object symbol to relations
                      # and belief computed by interpreting that single relation
        for rel in sg_dict["relations"]:
            object_symbol, landmark_symbol, best_match = rel
            if object_symbol not in oo_rels:
                oo_rels[object_symbol] = []

            keyword = best_match
            if keyword not in self._rules:
                print("Ignoring keyword '%s'. No rule provided." % keyword)
                continue
            if isinstance(self._rules[keyword], ForefRule)\
               and keyword not in foref_models:
                print("Ignoring keyword '%s'. ForefRule, but no model provided."
                      % keyword)
                continue
            print("Accepted keyword '%s'." % keyword)
            # Compute a belief distribution based on landmark locations
            # belief should be a mapping from (x,y) to a probability of object exist.
            kwargs = {}
            if isinstance(self._rules[keyword], ForefRule):
                # obtain frame of reference
                foref_model = foref_models[keyword]
                foref = foref_model(keyword, landmark_symbol, map_name,
                                    mapinfo, **foref_kwargs)
                if foref is None:
                    continue
                kwargs["foref"] = foref
                forefs_list.append((landmark_symbol, keyword, foref))

            landmark_locs = mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs
            belief = self._rules[keyword].compute(landmark_locs,
                                                  mapinfo.map_dims(map_name),
                                                  **kwargs)
            oo_rels[object_symbol].append((rel, belief))
            rules_used.append(keyword)

        # For each object, compute a distribution as a weighted sum of mixtures
        oo_belief = {}
        for object_symbol in oo_rels:
            if len(oo_rels[object_symbol]) == 0:
                print("Warning: Object {} has no relation".format(object_symbol))
                continue

            mixture_beliefs = []

            # mixture 0: joint everything
            joint_belief = None
            for rel, belief in oo_rels[object_symbol]:
                if joint_belief is None:
                    joint_belief = belief
                else:
                    joint_belief = combine_beliefs(belief, joint_belief, method="product")
            mixture_beliefs.append(joint_belief)

            # mixture 1: joint everything except for Foref relations
            joint_belief_no_foref = None
            for rel, belief in oo_rels[object_symbol]:
                keyword = rel[2]
                if isinstance(self._rules[keyword], ForefRule):
                    continue
                if joint_belief_no_foref is None:
                    joint_belief_no_foref = belief
                else:
                    joint_belief_no_foref = combine_beliefs(belief,
                                                            joint_belief_no_foref,
                                                            method="product")
            mixture_beliefs.append(joint_belief_no_foref)

            # mixture 2: at rule for all landmarks
            joint_belief_at = None
            for rel, _ in oo_rels[object_symbol]:
                landmark_symbol = rel[1]
                landmark_locs = mapinfo.landmark_footprint(landmark_symbol, map_name)
                belief = AtRule().compute(landmark_locs, mapinfo.map_dims(map_name))
                if joint_belief_at is None:
                    joint_belief_at = belief
                else:
                    joint_belief_at = combine_beliefs(joint_belief_at,
                                                      belief, method="add")
            mixture_beliefs.append(joint_belief_at)

            # mixture 3: near rule for all landmarks
            joint_belief_near = None
            for rel, _ in oo_rels[object_symbol]:
                landmark_symbol = rel[1]
                landmark_locs = mapinfo.landmark_footprint(landmark_symbol, map_name)
                belief = NearRule().compute(landmark_locs, mapinfo.map_dims(map_name))
                if joint_belief_near is None:
                    joint_belief_near = belief
                else:
                    joint_belief_near = combine_beliefs(joint_belief_near,
                                                        belief, method="add")
            mixture_beliefs.append(joint_belief_near)

            mixtures_used = []
            for mix_id in self.mixtures:
                mixtures_used.append(mixture_beliefs[mix_id])

            # weighted, combine
            weighted_sum_belief = {}
            total_prob = 0.0
            for loc in joint_belief:
                weighted_sum_belief[loc] = 0.0
                for i in range(len(mixtures_used)):
                    mixture_belief = mixtures_used[i]
                    if mixture_belief is None:
                        print("Skipped mixture %d"\
                              "(likely because the language contains only"\
                              "one foref-required predicate")
                        continue
                    weighted_sum_belief[loc] += self.weights[i] * mixture_belief[loc]
                total_prob += weighted_sum_belief[loc]
            for loc in weighted_sum_belief:
                weighted_sum_belief[loc] /= total_prob

            oo_belief[object_symbol] = weighted_sum_belief

        return oo_belief, {"landmark_footprints": landmark_footprints,
                           "rules_used": rules_used,
                           "sg_dict": sg_dict,
                           "forefs": forefs_list}


############# Keyword model ###########
class KeywordModel(InterpretModel):

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""

    def interpret(self, query, map_name, mapinfo,
                  symbol_to_synonyms=None,
                  spacy_model=None,
                  spatial_keywords=None,
                  allowed_keywords=None,
                  exact_matching=False):
        """Based on the given language and map (mapinfo),
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map.

        `allowed_keywords` (set) is used to make sure keywordModel is
        fair compared with rule based; Only for relations that contain
        the allowed_keywords, keyword rule will apply."""
        sg_dict = self._parse_query(query, map_name,
                                    spatial_keywords=spatial_keywords,
                                    symbol_to_synonyms=symbol_to_synonyms,
                                    spacy_model=spacy_model)
        oo_belief = {}  # mapping from object symbol to belief.
        landmark_footprints = {}
        for rel in sg_dict["relations"]:
            object_symbol, landmark_symbol, best_match = rel
            if allowed_keywords is not None and\
               best_match not in allowed_keywords:
                continue
            if exact_matching:
                # Match the landmark name exactly.
                landmark_name = mapinfo.name_for(landmark_symbol, map_name)
                if query.find(landmark_name) == -1:
                    continue
            landmark_locs = mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs
            w, l = mapinfo.map_dims(map_name)
            belief = {}
            for x in range(w):
                for y in range(l):
                    if (x,y) in landmark_locs:
                        belief[(x,y)] = 1
                    else:
                        belief[(x,y)] = 1e-9
            if object_symbol in oo_belief:
                oo_belief[object_symbol] =\
                    combine_beliefs(oo_belief[object_symbol],
                                    belief, method="add", normalize=False)
            else:
                oo_belief[object_symbol] = belief
        # Make everything uniform
        w, l = mapinfo.map_dims(map_name)
        for object_symbol in oo_belief:
            belief = oo_belief[object_symbol]
            # Belief is not normalized. So if there is a landmark at
            # this location, it will have a probability >= 1 (as assigned above).
            for loc in belief:
                if belief[loc] >= 1:
                    belief[loc] = 1
                else:
                    belief[loc] = 1.0 / (w*l)
        return oo_belief, {"landmark_footprints": landmark_footprints,
                           "sg_dict": sg_dict}


class GaussianPointModel(InterpretModel):
    def __init__(self, sigma):
        self.sigma = sigma

    def interpret(self, obj_poses, map_name, mapinfo):
        """For every object put a gaussian on top of the pose
        with variance `sigma`**2.

        `obj_poses` is a mapping from obj_symbol (e.g. RedHonda)
        to 2d pose (x,y)"""
        oo_belief = {}
        for object_symbol in obj_poses:
            pose = obj_poses[object_symbol]
            w, l = mapinfo.map_dims(map_name)
            belief = {}
            gauss = pomdp_py.Gaussian(list(pose),
                                      [[self.sigma**2, 0],
                                       [0, self.sigma**2]])
            # Normalize, cutoff, normalize
            total_prob = 0.0
            for x in range(w):
                for y in range(l):
                    belief[(x,y)] = gauss[(x,y)]
                    total_prob += belief[(x,y)]
            for loc in belief:
                belief[loc] = belief[loc] / total_prob
            # cutoff tails
            belief = cutoff(belief, (w,l))
            oo_belief[object_symbol] = belief
        return oo_belief

def evaluate(model, test_samples, mapinfo, keyword,
             map_dims=(41,41),
             foref_model=None,
             foref_model_path=None, device=None):
    """Evaluate the perplexity, kl divergence, distance between true / predicted
    object location, using rule based model and keyword model, given a dataset.

    test_samples is a list of tuples (obj_symbol, landmark_symbol, keyword, abs_obj_loc)
    """
    print("Loading symbol to synonyms...")
    with open(FILEPATHS["symbol_to_synonyms"]) as f:
        symbol_to_synonyms = json.load(f)
    print("Loading spatial keywords...")
    with open(FILEPATHS["relation_keywords"]) as f:
        spatial_keywords = json.load(f)

    # This is the foref model for the given keyword.
    foref_kwargs = {"device": device}
    if foref_model is None:
        if foref_model_path is not None:
            print("Loading pytorch model")
            nn_model = torch.load(foref_model_path)
            foref_model = nn_model.predict_foref

    all_locations = [(x,y)
                     for x in range(map_dims[0])
                     for y in range(map_dims[1])]

    results = {"perplex_true": [],  # The perplexity of a distribution for the true object location
               "perplex_pred": [],  # The perplexity of the predicted heatmap
               "kl_div": [],     # The kl divergence between true and predicted distributions
               "distance": []}  # The distance between most likely object location and true object location
    variance = [[1, 0], [0,1]]
    for i in range(len(test_samples)):
        obj_symbol, landmark, map_name, abs_obj_loc, map_img = test_samples[i]
        foref_kwargs["map_img"] = map_img
        true_dist = normal_pdf_2d(abs_obj_loc, variance, all_locations)
        oo_belief, meta = model.interpret(
            (obj_symbol, landmark, keyword),
            map_name, mapinfo,
            foref_models={keyword:foref_model},
            foref_kwargs={keyword:foref_kwargs},
            symbol_to_synonyms=symbol_to_synonyms,
            spatial_keywords=spatial_keywords)
        belief = oo_belief[obj_symbol]
        seqs, vals = dists_to_seqs([true_dist, belief], avoid_zero=True)
        perplex_true = perplexity(seqs[0])
        perplex_pred = perplexity(seqs[1])
        kl_div = kl_divergence(seqs[0], q=seqs[1])
        results["perplex_true"].append(perplex_true)
        results["perplex_pred"].append(perplex_pred)
        results["kl_div"].append(kl_div)

        objloc_pred = max(belief, key=lambda x: belief[x])
        dist = euclidean_dist(objloc_pred, abs_obj_loc)
        results["distance"].append(dist)
        sys.stdout.write("Computing heatmaps & metrics...[%d/%d]\r" % (i+1, len(test_samples)))

    results = compute_mean_ci(results)
    print("Summary results:")
    pprint(results["__summary__"])
    return results


def unittest():
    import matplotlib.pyplot as plt
    from sloop.models.heuristics.test import plot_belief
    from sloop.models.nn.plotting import plot_foref

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
    trial_path = "/home/kaiyuzh/repo/spatial-foref/sloop/"\
        "models/nn/logs/iter_1/context_foref/front/20200715092519288/"
    path_to_foref_model =\
        os.path.join(trial_path, "front_model.pt")
    normalizer_path = os.path.join(trial_path, "train_meta.json")
    with open(normalizer_path) as f:
        normalizers = json.load(f, object_hook=smart_json_hook)["_normalizers_"]
    print("Loading pytorch model")
    model = torch.load(path_to_foref_model)
    model.normalizers = normalizers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading spacy model...")
    spacy_model = spacy.load("en_core_web_md")

    # Use an example spatial graph
    # with open("./sg-example.json") as f:
    #     d = json.load(f)
    # sg = SpatialGraph.from_dict(d)
    # print(sg.to_dict())

    rules = {"near": NearRule(),
             "at": AtRule(),
             "beyond": BeyondRule(),
             "between": BetweenRule(),
             "east": DirectionRule("east"),
             "west": DirectionRule("west"),
             "north": DirectionRule("north"),
             "south": DirectionRule("south"),
             "front": ForefRule("front")}
    # rules = {"near": NearRule()}
    rbm = RuleBasedModel(rules)
    language = "The car is in front of Support Building."
    result = rbm.interpret(language, map_name, mapinfo,
                           spacy_model=spacy_model,
                           foref_model=model.predict_foref,
                           foref_kwargs={"device": device})
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
