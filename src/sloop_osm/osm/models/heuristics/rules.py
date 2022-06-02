# Defines a bunch of rules. Each rule has a `compute` function which
# returns the probability distribution of the predicate being satisfied
# given a landmark shape.

import math
import numpy as np
import scipy.stats
from sloop.datasets.utils import euclidean_dist

def combine_beliefs(belief1, belief2, method="product", normalize=True):
    result = dict(belief1)
    for loc in belief2:
        if loc not in result:
            result[loc] = belief2[loc]
        else:
            if method == "product":
                result[loc] *= belief2[loc]
            elif method == "add":
                result[loc] += belief2[loc]

    # Normalize
    if normalize:
        tot_prob = 0.0
        for loc in result:
            tot_prob += result[loc]
        for loc in result:
            result[loc] = result[loc] / tot_prob
    return result

def cutoff(belief, dim, landmarks={}, normalize=True, differentiate=False):
    """Cut off belief below the uniform threshold. Give a
    higher cut off belief for areas around mentioned landmarks."""
    width, length = dim
    total_prob = 0
    landmark_centers = []
    for symbol in landmarks:
        ctr = np.mean(landmarks[symbol], axis=0)
        radius = np.min(np.ptp(landmarks[symbol], axis=0)) / 2
        landmark_centers.append((ctr, radius))

    for loc in belief:
        if belief[loc] < 1.0 / (width*length):
            if differentiate:
                near_landmark = False
                for ctr, radius in landmark_centers:
                    if euclidean_dist(loc, ctr) < radius:
                        near_landmark = True
                if near_landmark:
                    belief[loc]  = 1e-9
                else:
                    belief[loc]  = 1e-12
            else:
                belief[loc]  = 1e-12
        total_prob += belief[loc]
    for loc in belief:
        belief[loc] = belief[loc] / total_prob
    return belief


############# Rules ###########
class Rule:
    def compute(self, landmark_locs, dims, **kwargs):
        raise NotImplementedError

class NearRule(Rule):
    """rule for 'near'"""
    def compute(self, landmark_locs, dims, field_width=3):
        # Refer to Maja Mataric's paper "Using Semantic Fields to Model Dynamic ..."
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        # This is computing the max of the span along each axis to get the
        # "width" of the landmark.
        center = np.mean(landmark_arr, axis=0)
        belief = {}
        tot_prob = 0
        for x in range(width):
            for y in range(length):
                # compute distance to nearest landmark location
                closest_lm_point = min(landmark_arr, key=lambda p: euclidean_dist(p, (x,y)))
                dist = euclidean_dist((x,y), closest_lm_point)
                belief[(x,y)] = math.exp(-(dist**2) / (2*field_width**2))
                tot_prob += belief[(x,y)]
        # Normalize
        for loc in belief:
            belief[loc] = belief[loc] / tot_prob
            if belief[loc] < 1.0 / (width*length):
                belief[loc] = 1e-9
        return belief

class AgainstRule(Rule):
    """rule for 'near'"""
    def compute(self, landmark_locs, dims):
        # Refer to Maja Mataric's paper "Using Semantic Fields to Model Dynamic ..."
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        # This is computing the max of the span along each axis to get the
        # "width" of the landmark.
        field_width = 1.5  # the field width here is 1 because we're treating
                         # each landmark point separately
        center = np.mean(landmark_arr, axis=0)
        belief = {}
        tot_prob = 0
        for x in range(width):
            for y in range(length):
                # compute distance to nearest landmark location
                closest_lm_point = min(landmark_arr, key=lambda p: euclidean_dist(p, (x,y)))
                dist = euclidean_dist((x,y), closest_lm_point)
                belief[(x,y)] = math.exp(-(dist**2) / (2*(1.5*field_width)**2)) - math.exp(-(dist**2) / (2*field_width**2))
                tot_prob += belief[(x,y)]
        # Normalize
        for loc in belief:
            belief[loc] = belief[loc] / tot_prob
        return belief

class BeyondRule(Rule):
    """rule for 'near'"""
    def compute(self, landmark_locs, dims):
        # Refer to Maja Mataric's paper "Using Semantic Fields to Model Dynamic ..."
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        # This is computing the max of the span along each axis to get the
        # "width" of the landmark.
        field_width = 1.5  # the field width here is 1 because we're treating
                         # each landmark point separately
        center = np.mean(landmark_arr, axis=0)
        belief = {}
        tot_prob = 0
        for x in range(width):
            for y in range(length):
                # compute distance to nearest landmark location
                closest_lm_point = min(landmark_arr, key=lambda p: euclidean_dist(p, (x,y)))
                dist = euclidean_dist((x,y), closest_lm_point)
                belief[(x,y)] = math.exp(-(dist**2) / (2*(3.0*field_width)**2)) - math.exp(-(dist**2) / (2*field_width**2))
                tot_prob += belief[(x,y)]
        # Normalize
        for loc in belief:
            belief[loc] = belief[loc] / tot_prob
        return belief


class AtRule(Rule):
    """rule for 'near'"""
    def compute(self, landmark_locs, dims):
        # Refer to Maja Mataric's paper "Using Semantic Fields to Model Dynamic ..."
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        # This is computing the max of the span along each axis to get the
        # "width" of the landmark.
        field_width = 1  # the field width here is 1 because we're treating
                         # each landmark point separately
        center = np.mean(landmark_arr, axis=0)
        belief = {}
        tot_prob = 0
        for x in range(width):
            for y in range(length):
                # compute distance to nearest landmark location
                closest_lm_point = min(landmark_arr, key=lambda p: euclidean_dist(p, (x,y)))
                dist = euclidean_dist((x,y), closest_lm_point)
                belief[(x,y)] = math.exp(-(dist**2) / (2*field_width**2))
                tot_prob += belief[(x,y)]
        # Normalize
        for loc in belief:
            belief[loc] = belief[loc] / tot_prob
        return belief

class BetweenRule(Rule):
    """rule for 'near'"""
    def __init__(self):
        self._near_rule = NearRule()
    def compute(self, landmark_locs, dims):
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        # This is computing the max of the span along each axis to get the
        # "width" of the landmark.
        center = np.mean(landmark_arr, axis=0)
        return self._near_rule.compute([center], dims, field_width=3)

class DirectionRule(Rule):
    """Direction rule is basically the same as FoRef rule, except
    with known frame of reference"""
    def __init__(self, direction):
        if direction.lower() not in {"north", "east", "west", "south",
                                     "northeast", "northwest",
                                     "southeast", "southwest"}:
            raise ValueError("Invalid direction %s for direction rule" % direction)
        self.direction = direction
        self.foref_rule = ForefRule("front")  # always front.

    def compute(self, landmark_locs, dims, field_width=None):
        center = np.mean(np.asarray(landmark_locs), axis=0)
        direction_to_angle = {"north": -90,
                              "south": 90,
                              "east": 0,
                              "west": 180,
                              "southeast": 45,
                              "southwest": 75,
                              "northeast": -45,
                              "northwest": 225}
        angle = math.radians(direction_to_angle[self.direction])
        foref = [*center, angle]
        return self.foref_rule.compute(landmark_locs, dims,
                                       foref, field_width=field_width)


class ForefRule(Rule):
    """Using frame of reference, predict what is front"""

    def __init__(self, direction):
        if direction not in {"front", "behind", "left", "right",
                             "above", "below"}:
            raise ValueError("Invalid direction %s for direction rule" % direction)
        self.direction = direction


    def compute(self, landmark_locs, dims, foref, field_width=None):
        """`foref` is the frame of reference (x,y,theta)."""
        width, length = dims
        landmark_arr = np.array(landmark_locs)
        fx, fy = foref[:2]
        fth = foref[2]

        if self.direction == "front":
            direction_vec = np.array([fx + math.cos(fth),
                                      fy + math.sin(fth)]) - np.array([fx, fy])
        elif self.direction == "behind":
            direction_vec = np.array([fx - math.cos(fth),
                                      fy - math.sin(fth)]) - np.array([fx, fy])
        elif self.direction == "right":
            direction_vec = np.array([fx + math.cos(fth + math.pi/2),
                                      fy + math.sin(fth + math.pi/2)]) - np.array([fx, fy])
        elif self.direction == "left":
            direction_vec = np.array([fx - math.cos(fth + math.pi/2),
                                      fy - math.sin(fth + math.pi/2)]) - np.array([fx, fy])

        center = np.mean(landmark_arr, axis=0)
        if field_width is None:
            field_width = min(4, max(1, np.max(np.ptp(landmark_arr, axis=0))))  # n,2
        belief = {}
        tot_prob = 0
        for x in range(width):
            for y in range(length):
                point_vec = np.array([x,y]) - np.array([fx, fy])
                if np.linalg.norm(point_vec) != 0:
                    point_vec /= np.linalg.norm(point_vec)

                closest_lm_point = min(landmark_arr, key=lambda p: euclidean_dist(p, (x,y)))
                dist = euclidean_dist((x,y), closest_lm_point)
                scale = np.dot(direction_vec, point_vec) * math.exp(-(dist**2) / (2*field_width**2))
                if scale > 0:
                    belief[(x,y)] = scale
                else:
                    belief[(x,y)] = 1e-9
                tot_prob += belief[(x,y)]
        # Normalize
        for loc in belief:
            belief[loc] = belief[loc] / tot_prob
        return belief


BASIC_RULES = {
    "above": DirectionRule("north"),
    "against": AgainstRule(),
    "along": AtRule(),
    "around": NearRule(),
    "below": DirectionRule("south"),
    "beside": NearRule(),
    "between": BetweenRule(),
    "down": DirectionRule("south"),
    "inside": AtRule(),
    "near": NearRule(),
    "off": AgainstRule(),
    "on": AtRule(),
    "by": AtRule(),
    "outside": NearRule(),
    "side": AtRule(),
    "under": DirectionRule("south"),
    "within": AtRule(),
    "top": DirectionRule("north"),
    "east": DirectionRule("east"),
    "west": DirectionRule("west"),
    "north": DirectionRule("north"),
    "south": DirectionRule("south"),
    "northwest": DirectionRule("northwest"),
    "southwest": DirectionRule("southwest"),
    "northeast": DirectionRule("northeast"),
    "southeast": DirectionRule("southeast"),
    "close": NearRule(),
    "at": AtRule(),
    "next": NearRule(),
    "middle": AtRule(),
    "towards": NearRule(),
    "adjacent": AtRule(),
    "in": AtRule(),
    "is": AtRule()
}
