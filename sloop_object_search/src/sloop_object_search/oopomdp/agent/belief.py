# utility functions for belief. The joint belief is modeled by
# pomdp_py.OOBelief
from tqdm import tqdm
import numpy as np
import pomdp_py
from ..domain.observation import RobotLocalization, RobotObservation
from ..models.search_region import SearchRegion2D, SearchRegion3D
from ..models.octree_belief import Octree, OctreeBelief, RegionalOctreeDistribution
from ..domain.state import ObjectState, RobotState
from sloop_object_search.utils.math import quat_to_euler, euler_to_quat, identity


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


##### Belief initialization ####
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
        object_belief_dist = {}

        object_loc_prior = {}
        for loc, prob in prior.get(objid, []):
            object_loc_prior[loc] = prob

        for loc in search_region:
            state = ObjectState(objid, target['class'], loc)
            # if prior is not empty, then for unspecified location,
            # the probability is zero
            if len(object_loc_prior) > 0:
                if loc in object_loc_prior:
                    object_belief_dist[state] = object_prior_dist[loc]
                else:
                    object_belief_dist[state] = 0.0
            else:
                # No prior specified, uniform.
                object_belief_dist[state] = 1.0 / len(search_region)

        object_beliefs[objid] = pomdp_py.Histogram(object_belief_dist)
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
