# ALL DEPRECATED. None of these is used in agent2.
print("ALL DEPRECATED. None of these is used in agent2")
import pomdp_py
from deprecated import deprecated
from tqdm import tqdm

from genmos_object_search.utils.math import normalize
from genmos_object_search.oopomdp.domain.observation import GMOSObservation
from genmos_object_search.oopomdp.models.transition_model import StaticObjectTransitionModel
from genmos_object_search.oopomdp.domain.state import (ObjectState,
                                                      RobotState)
from genmos_object_search.oopomdp.models.octree_belief import OctreeBelief, Octree
from genmos_object_search.oopomdp.deprecated.domain.state import RobotStateTopo
from sloop.observation import SpatialLanguageObservation

##################### Belief 2D ##########################
class Belief2D(pomdp_py.OOBelief):

    @staticmethod
    def init_object_belief(objid, objclass, search_region, prior=None):
        if prior is None:
            prior = {}
        belief_dist = {}
        object_prior_dist = prior.get(objid, {})
        if type(object_prior_dist) != dict:
            object_prior_dist = {}
        for loc in search_region:
            state = ObjectState(objid, objclass, loc)
            if loc in object_prior_dist:
                belief_dist[state] = object_prior_dist[loc]
            else:
                # uniform
                belief_dist[state] = 1.0 / len(search_region)
        return pomdp_py.Histogram(belief_dist)

    @staticmethod
    def init_object_beliefs(target_objects, search_region, prior=None):
        """prior: dictionary objid->{loc->prob}"""
        object_beliefs = {}
        if prior is None:
            prior = {}
        for objid in target_objects:
            target = target_objects[objid]
            object_beliefs[objid] = Belief2D.init_object_belief(
                objid, target['class'], search_region, prior=prior)
        return object_beliefs

    def __init__(self,
                 target_objects,
                 robot_state=None,
                 robot_belief=None,
                 belief_config=None,
                 search_region=None,
                 object_beliefs=None):
        """Note that object_beliefs don't include robot belief"""
        self.robot_id = robot_state["id"]
        self.search_region = search_region
        self.target_objects = target_objects
        if robot_belief is None:
            assert robot_state is not None,\
                "either 'robot_belief' or 'robot_state' is needed"
            robot_belief = pomdp_py.Histogram({robot_state:1.0})
        robot_id = robot_belief.random()["id"]
        if object_beliefs is not None:
            object_beliefs[robot_id] = robot_belief

        else:
            assert search_region is not None,\
                "search region is required to initialize belief"
            prior = belief_config.get("prior", {})
            if type(prior) != dict:
                prior = {}

            object_beliefs = Belief2D.init_object_beliefs(
                target_objects, search_region, prior=prior)
            object_beliefs[robot_id] = robot_belief
        super().__init__(object_beliefs)

    @deprecated(reason="Not used in agent2. Belief update is handled by the agent.")
    def update_object_belief(self, agent, objid, observation,
                             next_robot_state, action,
                             observation_model=None):
        """
        will update the belief with the given observation_model
        (default its the agent's built-in; But it may be set to the
        spatial language observation model, which is separate
        from the agent's model for planning.
        """
        if isinstance(observation, GMOSObservation):
            zobj = observation.z(objid)
        elif isinstance(observation, SpatialLanguageObservation):
            zobj = observation
        else:
            raise TypeError(f"observation type {observation.__class__.__name__} unexpected")

        assert isinstance(agent.transition_model.transition_models[objid],
                          StaticObjectTransitionModel)
        belief_dist = {}
        for loc in self.search_region:
            objclass = self.target_objects[objid]["class"]
            object_state = ObjectState(objid, objclass, loc)
            snext = pomdp_py.OOState({
                objid: object_state,
                self.robot_id: next_robot_state
            })
            if observation_model is None:
                observation_model = agent.observation_model[objid]
            pr_z = observation_model.probability(zobj, snext, action)
            belief_dist[object_state] = pr_z * self.b(objid)[object_state]

        belief_dist = pomdp_py.Histogram(normalize(belief_dist))
        self.set_object_belief(objid, belief_dist)

# Deprecated; not used in agent2
class BeliefBasic2D(Belief2D):

    @deprecated(reason="Not used in agent2. Belief update is handled by the agent.")
    def update_robot_belief(self, observation, action):
        # Note: assumes robot state observable
        if isinstance(observation, SpatialLanguageObservation):
            # spatial language doesn't involve robot
            return

        zrobot = observation.z(self.robot_id)
        next_robot_state = RobotState(zrobot.robot_id,
                                      zrobot.pose,
                                      zrobot.objects_found,
                                      zrobot.camera_direction)
        self.set_object_belief(
            self.robot_id, pomdp_py.Histogram({next_robot_state: 1.0}))

# Deprecated; not used in agent2
class BeliefTopo2D(Belief2D):

    @deprecated(reason="Not used in agent2. Belief update is handled by the agent.")
    def update_robot_belief(self, observation, action):
        # Note: assumes robot state observable
        if isinstance(observation, SpatialLanguageObservation):
            # spatial language doesn't involve robot
            return

        if self.robot_id not in observation:
            # we don't have observation about the robot; so skip
            return

        zrobot = observation.z(self.robot_id)
        next_robot_state = RobotStateTopo(zrobot.robot_id,
                                          zrobot.pose,
                                          zrobot.objects_found,
                                          zrobot.camera_direction,
                                          zrobot.topo_nid)
        self.set_object_belief(
            self.robot_id, pomdp_py.Histogram({next_robot_state: 1.0}))


    @staticmethod
    def combine_object_beliefs(search_region,
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
##################### Belief 3D ##########################
class BeliefBasic3D(pomdp_py.OOBelief):
    def __init__(self, robot_state, target_objects, belief_config):
        robot_belief = pomdp_py.Histogram({robot_state:1.0})

        # Super basic for now. No consideration of prior, or
        # the size and shape of the search region. TODO.
        object_beliefs = {robot_state["id"]: robot_belief}
        for objid in target_objects:
            target = target_objects[objid]
            octree = Octree((16, 16, 16))
            object_beliefs[objid] = OctreeBelief(16, 16, 16,
                                                 objid, target["class"], octree)
        super().__init__(object_beliefs)
