# utility functions for belief. The joint belief is modeled by
# pomdp_py.OOBelief
from tqdm import tqdm
import pomdp_py
from ..models.search_region import SearchRegion2D, SearchRegion3D
from ..domain.state import ObjectState, RobotState

def init_robot_belief(robot_config, robot_pose_dist, robot_state_class=RobotState, **state_kwargs):
    """Given a distribution of robot pose, create a belief over
    robot state with the same representation as that distribution."""
    if isinstance(robot_pose_dist, pomdp_py.WeightedParticles):
        robot_state_particles = []
        for pose, weight in robot_pose_dist.particles:
            assert type(pose) == tuple, "pose should be a tuple"
            robot_state = robot_state_class(
                robot_config["id"], pose, (), None, **state_kwargs)
            robot_state_particles.append((robot_state, weight))
        return pomdp_py.WeightedParticles(robot_state_particles)
    elif isinstance(robot_pose_dist, pomdp_py.Histogram):
        robot_state_dist = {}
        for pose in robot_pose_dist:
            assert type(pose) == tuple, "pose should be a tuple"
            robot_state = robot_state_class(
                robot_config["id"], pose, (), None, **state_kwargs)
            robot_state_dist[robot_state] = robot_pose_dist[pose]
        return pomdp_py.Histogram(robot_state_dist)
    else:
        raise NotImplementedError("currently, {type(robot_pose_dist)}"
                                  "is not a supported type of robot pose distribution.")

def init_object_beliefs_2d(target_objects, search_region, prior=None):
    """prior: dictionary objid->{loc->prob}"""
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

def init_object_beliefs(target_objects, search_region, prior=None):
    if isinstance(search_region, SearchRegion2D):
        return init_object_beliefs_2d(target_objects, search_region, prior=prior)
    else:
        assert isinstance(search_region, SearchRegion3D),\
            "search region is of invalid type ({}).".format(type(search_region))
        raise NotImplementedError()

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
