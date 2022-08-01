import pomdp_py
from tqdm import tqdm
from .transition_model import StaticObjectTransitionModel
from ..domain.state import (ObjectState,
                            RobotState,
                            RobotStateTopo)
from sloop_object_search.utils.math import normalize
from sloop_object_search.oopomdp.domain.observation import GMOSObservation
from sloop.observation import SpatialLanguageObservation


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

    def __init__(self,
                 robot_state,
                 target_objects,
                 belief_config=None,
                 search_region=None,
                 object_beliefs=None):
        """Note that object_beliefs don't include robot belief"""
        self.robot_id = robot_state["id"]
        self.search_region = search_region
        self.target_objects = target_objects
        robot_belief = pomdp_py.Histogram({robot_state:1.0})
        if object_beliefs is not None:
            object_beliefs[robot_state["id"]] = robot_belief
        else:
            prior = belief_config.get("prior", {})
            if type(prior) != dict:
                prior = {}

            object_beliefs = {robot_state["id"]: robot_belief}
            for objid in target_objects:
                target = target_objects[objid]
                object_beliefs[objid] = Belief2D.init_object_belief(
                    objid, target['class'], search_region, prior=prior)
        super().__init__(object_beliefs)


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


class BeliefBasic2D(Belief2D):
    def update_robot_belief(self, observation, action):
        # Note: assumes robot state observable
        if isinstance(observation, SpatialLanguageObservation):
            # spatial language doesn't involve robot
            return

        next_robot_state = RobotState.from_obz(observation.z(self.robot_id))
        self.set_object_belief(
            self.robot_id, pomdp_py.Histogram({next_robot_state: 1.0}))


class BeliefTopo2D(Belief2D):

    def update_robot_belief(self, observation, action):
        # Note: assumes robot state observable
        if isinstance(observation, SpatialLanguageObservation):
            # spatial language doesn't involve robot
            return

        if self.robot_id not in observation:
            # we don't have observation about the robot; so skip
            return

        next_robot_state = RobotStateTopo.from_obz(observation.z(self.robot_id))
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
