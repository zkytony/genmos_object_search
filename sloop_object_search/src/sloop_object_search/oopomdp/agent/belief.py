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

class RobotPoseDist(pomdp_py.Gaussian):
    """This wraps pomdp_py Gaussian as the belief
    over robot pose. Note that we represent robot pose in 3D
    as (x, y, z, qx, qy, qz, qw), while its covariance
    matrix is about the rotations around the axes. THis
    wrapper handles the conversion."""
    def __init__(self, pose, covariance):
        """
        covariance (array or list): covariance matrix.
            For 3D pose, the variables are [x y z thx thy thz]
            For 2D pose, the variables are [x y yaw]
        """
        if len(pose) == 7:
            self.is_3d = True
        elif len(pose) == 3:
            self.is_3d = False
        else:
            raise ValueError("Invalid dimension of robot pose estimate. "\
                             f"Expected 3 or 7, but got {len(pose)}")

        if isinstance(covariance, np.ndarray):
            covariance = covariance.tolist()
        if self.is_3d:
            super().__init__(list(RobotPoseDist._to_euler_pose(pose)), covariance)
        else:
            super().__init__(list(pose), covariance)

    @staticmethod
    def _to_euler_pose(pose7tup):
        return (*pose7tup[:3], *quat_to_euler(*pose7tup[3:]))

    @staticmethod
    def _to_quat_pose(pose6tup):
        return (*pose6tup[:3], *euler_to_quat(*pose6tup[3:]))

    def __getitem__(self, pose):
        if self.is_3d:
            return super().__getitem__(RobotPoseDist._to_euler_pose(pose))
        else:
            return super().__getitem__(pose)

    def __mpe__(self):
        pose = super().mpe()
        if self.is_3d:
            return RobotPoseDist._to_quat_pose(pose)
        else:
            return pose

    def random(self):
        pose = super().random()
        if self.is_3d:
            return RobotPoseDist._to_quat_pose(pose)
        else:
            return pose


class RobotStateBelief(pomdp_py.GenerativeDistribution):
    """This is a distribution that samples RobotState obejcts"""
    def __init__(self, robot_id, pose_dist, robot_state_class=RobotState,
                 epsilon=1e-12, **state_kwargs):
        self.robot_id = robot_id
        self.pose_dist = pose_dist
        self.robot_state_class = robot_state_class
        self.objects_found = state_kwargs.pop("objects_found", ())
        self.camera_direction = state_kwargs.pop("camera_direction", None)
        self.state_kwargs = state_kwargs
        self.epsilon = epsilon


    def random(self):
        robot_pose = self.pose_dist.random()
        return self.robot_state_class(self.robot_id, robot_pose,
                                      self.objects_found,
                                      self.camera_direction,
                                      **self.state_kwargs)

    def mpe(self):
        robot_pose = self.pose_dist.random()
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
            self.pose_dist[srobot.pose]


##### Belief initialization ####
def robot_pose_dist_from_localization(robot_localization):
    """
    Args:
        robot_localization (RobotLocalization)
    """
    return RobotPoseDist(robot_localization.pose, robot_localization.cov)

def init_robot_belief(robot_config, robot_pose_dist, robot_state_class=RobotState, **state_kwargs):
    """Given a distribution of robot pose, create a belief over
    robot state with the same representation as that distribution."""
    if not isinstance(robot_pose_dist, RobotPoseDist):
        raise NotImplementedError("{type(robot_pose_dist)}"
                                  " is not a supported type of robot pose distribution.")
    return RobotStateBelief(robot_config["id"],
                            robot_pose_dist,
                            robot_state_class,
                            objects_found=(),
                            camera_direction=None,
                            **state_kwargs)

def init_object_beliefs_2d(target_objects, search_region, belief_config={}):
    """prior: dictionary objid->{loc->prob}"""
    assert isinstance(search_region, SearchRegion2D),\
        f"search_region should be a SearchRegion2D but its {type(search_region)}"
    prior = belief_config.get("prior", {})
    object_beliefs = {}
    if prior is None:
        prior = {}
    for objid in target_objects:
        target = target_objects[objid]
        object_prior = prior.get(objid, {})
        object_belief_dist = {}
        for loc in search_region:
            state = ObjectState(objid, target['class'], loc)
            if object_prior == "uniform":
                # uniform
                object_belief_dist[state] = 1.0 / len(search_region)
            else:
                if loc in object_prior:
                    object_belief_dist[state] = object_prior_dist[loc]
                else:
                    # unspecified, still uniform
                    object_belief_dist[state] = 1.0 / len(search_region)
        object_beliefs[objid] = pomdp_py.Histogram(object_belief_dist)
    return object_beliefs

def init_object_beliefs_3d(target_objects, search_region, belief_config={}):
    """we'll use Octree belief. Here, 'search_region' should be a
    SearchRegion3D. As such, it has an RegionalOctreeDistribution
    used for modeling occupancy. The agent's octree belief will be
    based on that.
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
            for x,y,z,r in prior[objid]:
                state = ObjectState(objid, target["class"], (x,y,z), res=r)
                octree_belief.assign(state, prior[objid][(x,y,z,r)])
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


#### Belief Update ####
def get_zrobot_for_update(observation):
    if isinstance(observation, JointObservation):
        zrobot = observation.z(self.robot_id)
    else:
        zrobot = observation

    # We need to work with RobotObservation objects
    if isinstance(zrobot, RobotLocalization):
        robot_state = current_robot_belief.mpe()
        zrobot = robot_observation_model.observation_class.from_state(robot_state, pose=zrobot.pose)
    else:
        assert isinstance(zrobot, RobotObservation),\
            f"failed to extract robot observation from given observation of type {type(observation)}"
    return zrobot


def update_robot_belief(current_robot_belief, observation, agent):
    """"""
    zrobot = get_zrobot_for_update(observation)



    if isinstance(zrobot, RobotLocalization):
        # change zrobot to be of type compatible with robot_observation_model
        robot_state = current_robot_belief.mpe()
        zrobot = robot_observation_model.observation_class.from_state(robot_state, pose=zrobot.pose)

    if isinstance(current_robot_belief, pomdp_py.WeightedParticles):
        raise NotImplementedError()



    if isinstance(obseration, RobotLocalization):
        pass

def reinvigorate_robot_belief_particles(zrobot, srobot_from_zrobot_func, num_particles=100):
    """srobot_from_zrobot_func is a function that takes in a RobotObservation
    and returns a RobotState (could be their children classes)"""
    raise NotImplementedError()
