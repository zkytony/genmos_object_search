import json
import spacy
import os, sys
import torch
import pomdp_py
from pprint import pprint

from sloop.observation_model import SpatialLanguageObservationModel

from sloop.osm.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.osm.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.osm.datasets.utils import make_context_img
from sloop.osm.datasets.dataloader import *
from sloop.utils import smart_json_hook
from ..nn.metrics import *
from .rules import *

############# Rule based model ###########
class RuleBasedModel(SpatialLanguageObservationModel):
    # Simple product SLU model

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""

    def __init__(self, rules, mapinfo, foref_models={}, foref_kwargs={}, **kwargs):
        """
        `rules` is a mapping from keyword to a Rule
        """
        self._rules = rules
        self._mapinfo = mapinfo
        self._foref_models = foref_models
        self._foref_kwargs = foref_kwargs
        super().__init__(**kwargs)

    def interpret(self, splang_observation):
        """Given a spatial language observation (splang_observation)
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map.
        """
        oo_belief = {}  # mapping from object symbol to belief.
        landmark_footprints = {}
        rules_used = []
        forefs_list = []  # list of (landmark, keyword, frame of reference)
        for rel in splang_observation:
            object_symbol, landmark_symbol, best_match = rel
            landmark_locs = self._mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs

            for keyword in [best_match]:  # TODO: REFACTOR
                if keyword not in self._rules:
                    if keyword == "front":
                        import pdb; pdb.set_trace()
                    print("Ignoring keyword '%s'. No rule provided." % keyword)
                    continue
                if isinstance(self._rules[keyword], ForefRule)\
                   and keyword not in self._foref_models:
                    print("Ignoring keyword '%s'. ForefRule, but no model provided."
                          % keyword)
                    continue
                print("Accepted keyword '%s'." % keyword)
                # Compute a belief distribution based on landmark locations
                # belief should be a mapping from (x,y) to a probability of object exist.
                kwargs = {}
                if isinstance(self._rules[keyword], ForefRule):
                    # obtain frame of reference
                    foref_model = self._foref_models[keyword]
                    foref = foref_model(keyword, landmark_symbol, map_name,
                                        self._mapinfo, **self._foref_kwargs)
                    if foref is None:
                        continue
                    kwargs["foref"] = foref
                    forefs_list.append((landmark_symbol, keyword, foref))
                belief = self._rules[keyword].compute(landmark_locs,
                                                      self._mapinfo.map_dims(map_name),
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
            oo_belief[object_symbol] = cutoff(belief, self._mapinfo.map_dims(map_name), landmark_footprints)
        return oo_belief, {"landmark_footprints": landmark_footprints,
                           "rules_used": rules_used,
                           "forefs": forefs_list}

############# Multi(level/interpretation) model ###########
class MixtureSLUModel(SpatialLanguageObservationModel):

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""

    def __init__(self, rules, mapinfo, mixtures=[0,1,2,3], weights=[0.6, 0.25, 0.1, 0.05],
                 foref_models={}, foref_kwargs={}, **kwargs):
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
        self._mapinfo = mapinfo
        self.mixtures = mixtures
        self.weights = weights
        self._foref_models = foref_models
        self._foref_kwargs = foref_kwargs
        super().__init__(**kwargs)

    def interpret(self, splang_observation):
        # The parse query here also matches keyword
        map_name = splang_observation.map_name
        landmark_footprints = {}
        rules_used = []
        forefs_list = []
        oo_rels = {}  # mapping from object symbol to relations
                      # and belief computed by interpreting that single relation
        for rel in splang_observation:
            object_symbol, landmark_symbol, best_match = rel
            if object_symbol not in oo_rels:
                oo_rels[object_symbol] = []

            keyword = best_match
            if keyword not in self._rules:
                print("Ignoring keyword '%s'. No rule provided." % keyword)
                continue
            if isinstance(self._rules[keyword], ForefRule)\
               and keyword not in self._foref_models:
                print("Ignoring keyword '%s'. ForefRule, but no model provided."
                      % keyword)
                continue
            print("Accepted keyword '%s'." % keyword)
            # Compute a belief distribution based on landmark locations
            # belief should be a mapping from (x,y) to a probability of object exist.
            kwargs = {}
            if isinstance(self._rules[keyword], ForefRule):
                # obtain frame of reference
                foref_model = self._foref_models[keyword]
                foref = foref_model(keyword, landmark_symbol, map_name,
                                    self._mapinfo, **self._foref_kwargs)
                if foref is None:
                    continue
                kwargs["foref"] = foref
                forefs_list.append((landmark_symbol, keyword, foref))

            landmark_locs = self._mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs
            belief = self._rules[keyword].compute(landmark_locs,
                                                  self._mapinfo.map_dims(map_name),
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
                landmark_locs = self._mapinfo.landmark_footprint(landmark_symbol, map_name)
                belief = AtRule().compute(landmark_locs, self._mapinfo.map_dims(map_name))
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
                landmark_locs = self._mapinfo.landmark_footprint(landmark_symbol, map_name)
                belief = NearRule().compute(landmark_locs, self._mapinfo.map_dims(map_name))
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
                           "forefs": forefs_list}


############# Keyword model ###########
class KeywordModel(SpatialLanguageObservationModel):

    """A rule based model will output belief distribution
    from a language input based on manually coded rules;
    We follow the research by John O'Keefe on Place Fields."""
    def __init__(self, mapinfo):
        self._mapinfo = mapinfo

    def interpret(self, splang_observation):
        """Based on the given language and map (mapinfo),
        return a matrix belief distribution of the objects
        mentioned in the language on top of the map.

        `allowed_keywords` (set) is used to make sure keywordModel is
        fair compared with rule based; Only for relations that contain
        the allowed_keywords, keyword rule will apply."""
        map_name = splang_observation.map_name
        oo_belief = {}  # mapping from object symbol to belief.
        landmark_footprints = {}
        for rel in splang_observation:
            object_symbol, landmark_symbol, best_match = rel
            if allowed_keywords is not None and\
               best_match not in allowed_keywords:
                continue
            if exact_matching:
                # Match the landmark name exactly.
                landmark_name = self._mapinfo.name_for(landmark_symbol, map_name)
                if query.find(landmark_name) == -1:
                    continue
            landmark_locs = self._mapinfo.landmark_footprint(landmark_symbol, map_name)
            landmark_footprints[landmark_symbol] = landmark_locs
            w, l = self._mapinfo.map_dims(map_name)
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
        w, l = self._mapinfo.map_dims(map_name)
        for object_symbol in oo_belief:
            belief = oo_belief[object_symbol]
            # Belief is not normalized. So if there is a landmark at
            # this location, it will have a probability >= 1 (as assigned above).
            for loc in belief:
                if belief[loc] >= 1:
                    belief[loc] = 1
                else:
                    belief[loc] = 1.0 / (w*l)
        return oo_belief, {"landmark_footprints": landmark_footprints}


class GaussianPointModel(SpatialLanguageObservationModel):
    def __init__(self, sigma):
        self.sigma = sigma

    def probability(self, splang_observation, next_state, objid=None):
        raise NotImplementedError("Spatial language observation is"
                                  "not considered by GaussianPointModel."
                                  "Directly call 'interpret' instead.")

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
