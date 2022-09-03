# utility functions for belief. The joint belief is modeled by
# pomdp_py.OOBelief
from tqdm import tqdm
import numpy as np
import pomdp_py
import random
from ..domain.observation import RobotLocalization, RobotObservation, FovVoxels, Voxel
from ..models.search_region import SearchRegion2D, SearchRegion3D
from ..models.octree_belief import Octree, OctreeBelief, RegionalOctreeDistribution
from ..domain.state import ObjectState, RobotState
from sloop_object_search.utils.math import quat_to_euler, euler_to_quat, identity


##### Belief initialization utilities ####
def init_robot_belief(robot_config, robot_pose_est, robot_state_class=RobotState, **state_kwargs):
    """Given a distribution of robot pose, create a belief over
    robot state with the same representation as that distribution."""
    if not isinstance(robot_pose_est, RobotLocalization):
        raise NotImplementedError("{type(robot_pose_est)}"
                                  " is not a supported type of robot pose distribution.")
    return RobotStateBelief(robot_config["id"],
                            robot_pose_est,
                            robot_state_class,
                            objects_found=(),
                            camera_direction=None,
                            **state_kwargs)

def init_object_beliefs_2d(target_objects, search_region, belief_config={}):
    """prior: dictionary objid->[[loc, prob]]"""
    assert isinstance(search_region, SearchRegion2D),\
        f"search_region should be a SearchRegion2D but its {type(search_region)}"
    prior = belief_config.get("prior", {})
    object_beliefs = {}
    if prior is None:
        prior = {}
    for objid in target_objects:
        target = target_objects[objid]
        loc_dist = LocDist2D(search_region)
        object_belief = ObjectBelief2D(objid, target['class'], loc_dist)
        for loc, prob in prior.get(objid, []):
            state = ObjectState(objid, target["class"], loc)
            # TODO: make 'normalized' configurable
            object_belief.assign(state, prob, normalized=True)
        object_beliefs[objid] = object_belief
    return object_beliefs

def init_object_beliefs_3d(target_objects, search_region, belief_config={}):
    """we'll use Octree belief. Here, 'search_region' should be a
    SearchRegion3D. As such, it has an RegionalOctreeDistribution
    used for modeling occupancy. The agent's octree belief will be
    based on that.

    prior in belief_config is a dictionary {objid->(loc,prob)}
    """
    assert isinstance(search_region, SearchRegion3D),\
        f"search_region should be a SearchRegion3D but its {type(search_region)}"
    prior = belief_config.get("prior", {})
    init_params = belief_config.get("init_params", {})
    object_beliefs = {}
    dimension = search_region.octree_dist.octree.dimensions[0]
    for objid in target_objects:
        target = target_objects[objid]
        octree_dist = RegionalOctreeDistribution(
            (dimension, dimension, dimension),
            search_region.octree_dist.region,
            num_samples=init_params.get("num_samples", 200))
        octree_belief = OctreeBelief(objid, target['class'], octree_dist)
        if prior is not None and objid in prior:
            for voxel, prob in prior[objid]:
                x,y,z,r = voxel
                state = ObjectState(objid, target["class"], (x,y,z), res=r)
                # TODO: make 'normalized' configurable
                octree_belief.assign(state, prob, normalized=True)
        object_beliefs[objid] = octree_belief
    return object_beliefs

def init_object_beliefs(target_objects, search_region, belief_config={}):
    if isinstance(search_region, SearchRegion2D):
        return init_object_beliefs_2d(target_objects, search_region, belief_config=belief_config)
    else:
        assert isinstance(search_region, SearchRegion3D),\
            "search region is of invalid type ({}).".format(type(search_region))
        return init_object_beliefs_3d(target_objects, search_region, belief_config=belief_config)

def accumulate_object_beliefs(search_region,
                              object_beliefs):
    """
    Given object_beliefs (dict, mapping from objid to belief histogram),
    returns a mapping from location to probability where the probability
    is the result of adding up probabilities from different objects at
    the same location.

    Args:
        search_region (set): set of locations
        object_beliefs (dict): mapping from objid to belief histogram
    Returns:
        dict
    """
    dist = {}
    for loc in tqdm(search_region):
        dist[loc] = 1e-9
        for objid in object_beliefs:
            random_sobj = object_beliefs[objid].random()
            sobj = ObjectState(objid, random_sobj.objclass, loc)
            dist[loc] += object_beliefs[objid][sobj]
    return dist


class RobotStateBelief(pomdp_py.GenerativeDistribution):
    """This is a distribution that samples RobotState obejcts"""
    def __init__(self, robot_id, pose_est, robot_state_class=RobotState,
                 epsilon=1e-12, **state_kwargs):
        if not isinstance(pose_est, RobotLocalization):
            raise TypeError("pose_est should be a RobotLocalization")
        self.robot_id = robot_id
        self.pose_est = pose_est
        self.robot_state_class = robot_state_class
        self.objects_found = state_kwargs.pop("objects_found", ())
        self.camera_direction = state_kwargs.pop("camera_direction", None)
        self.state_kwargs = state_kwargs
        self.epsilon = epsilon


    def random(self):
        robot_pose = self.pose_est.random()
        return self.robot_state_class(self.robot_id, robot_pose,
                                      self.objects_found,
                                      self.camera_direction,
                                      **self.state_kwargs)

    def mpe(self):
        robot_pose = self.pose_est.random()
        return self.robot_state_class(self.robot_id, robot_pose,
                                      self.objects_found,
                                      self.camera_direction,
                                      **self.state_kwargs)

    def __getitem__(self, srobot):
        if not isinstance(srobot, self.robot_state_class):
            raise TypeError(f"srobot is not of accepted robot class {self.robot_state_class}")
        srobot_mimic = self.robot_state_class(self.robot_id,
                                              srobot.pose,
                                              self.objects_found,
                                              self.camera_direction,
                                              **self.state_kwargs)
        if srobot_mimic != srobot:
            return self.epsilon
        else:
            self.pose_est[srobot.pose]


class LocDist2D(pomdp_py.GenerativeDistribution):
    """This is the belief representation over an object's location in 2D. Similar to
    Octree Belief, this representation maintains a collection of locations, each
    with an unnormalized probability. A separate normalizer is maintained. This
    representation enables updating belief by only updating the values in a
    small region, without the need to enumerate over the entire space. But unlike
    octree belief, this representation initializes explicitly a value for every
    location, because we assume it is efficient enough computationally to do so for 2D.
    """
    def __init__(self, search_region, default_val=1.):
        """
        belief_config: the 'belief' dict in agent_config"""
        assert isinstance(search_region, SearchRegion2D),\
            f"search_region should be a SearchRegion2D but its {type(search_region)}"
        # maps from a location in search region to an unnormalized probability value
        self._search_region = search_region
        self._default_val = default_val
        self._values = {loc: self._default_val
                       for loc in self._search_region}
        self._normalizer = len(self._search_region) * self._default_val

    def prob_at(self, x, y):
        if (x,y) not in self._search_region:
            return 0.0
        else:
            return self._values[(x,y)] / self._normalizer

    def get_val(self, x, y):
        return self._values[(x,y)]

    def in_region(self, pos):
        return pos in self._values

    def assign(self, pos, value, normalized=False):
        """Sets the value at a position to be the given. If 'normalized' is True, then
        'value' is a normalized probability.
        """
        if type(pos) != tuple and len(pos) != 2:
            raise ValueError("Requires pos to be a tuple (x,y)")
        if pos not in self._search_region:
            raise ValueError(f"pos {pos} is not in search region")
        if normalized:
            prob = value
            val_pos = self.prob_at(*pos) * self._normalizer
            dv = (prob*self._normalizer - val_pos) / (1.0 - prob)
            value = val_pos + dv
        old_val = self._values[pos]
        self._values[pos] = value
        self.update_normalizer(old_val, value)

    def update_normalizer(self, old_value, value):
        self._normalizer += (value - old_value)

    def __getitem__(self, pos):
        if type(pos) != tuple and len(pos) != 2:
            raise ValueError("Requires pos to be a tuple (x,y)")
        return self.prob_at(*pos)

    def __setitem__(self, pos, value):
        self.assign(pos, value)

    def __iter__(self):
        return iter(self._values)

    def random(self, rnd=random):
        candidates = list(self._values.keys())
        weights = [self._values[loc] for loc in candidates]
        return rnd.choices(candidates, weights=weights, k=1)[0]

    def mpe(self):
        return max(self._values, key=self._values.get)

    def prob_in_rect(self, center, w, l):
        cx, cy = map(lambda v: int(round(v)), center)
        total_prob = 0.0
        for x in range(cx - w//2, cx + w//2):
            for y in range(cy - l//2, cy + l//2):
                if (x,y) in self._search_region:
                    total_prob += self.prob_at(x,y)
        return total_prob


class ObjectBelief2D(pomdp_py.GenerativeDistribution):
    """A wrapper around LocDist2D that takes in and outputs ObjectState"""
    def __init__(self, objid, objclass, loc_dist):
        if not isinstance(loc_dist, LocDist2D):
            raise TypeError("loc_dist must be an instance of LocDist2D")
        self._loc_dist = loc_dist
        self._objid = objid
        self._objclass = objclass

    @property
    def objid(self):
        return self._objid

    @property
    def loc_dist(self):
        return self._loc_dist

    def __getitem__(self, object_state):
        if object_state.id != self._objid:
            raise TypeError("Given object state has object id %d\n"\
                            "but this belief is for object id %d"
                            % (object_state.id, self._objid))
        x, y = object_state.loc
        return self._loc_dist.prob_at(x, y)

    def __setitem__(self, object_state, value):
        x, y = object_state.loc
        self._loc_dist[(x,y)] = value

    def assign(self, object_state, value, normalized=False):
        x,y = object_state.loc
        self._loc_dist.assign((x,y), value,
                              normalized=normalized)

    def random(self):
        pos = self._loc_dist.random()
        return ObjectState(self._objid, self._objclass, pos)

    def mpe(self):
        pos = self._loc_dist.mpe()
        return ObjectState(self._objid, self._objclass, pos)


def update_object_belief_2d(object_belief_2d, real_observation,
                            alpha=1000., beta=0., gamma=1.0):
    """real_observation should be a FovVoxels; update in place"""
    if not type(real_observation) == set:
       raise TypeError("Belief update should happen using"\
                       " cells in the FOV (type set)")

    for loc, label in real_observation:
        if not object_belief_2d.loc_dist.in_region(loc):
            continue
        val_t = object_belief_2d.loc_dist.get_val(*loc)
        if label == "free":
            val_tp1 = val_t * beta
        elif label == object_belief_2d.objid:
            val_tp1 = val_t * alpha
        else:
            raise ValueError(f"Unexpected label {label}")
        object_belief_2d.loc_dist[loc] = val_tp1
        object_belief_2d.loc_dist.update_normalizer(val_t, val_tp1)
    return object_belief_2d
