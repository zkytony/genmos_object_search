import pomdp_py
from .transition_model import StaticObjectTransitionModel
from ..domain.state import ObjectState2D, RobotState2D
from sloop_object_search.utils.math import normalize


class BeliefBasic2D(pomdp_py.OOBelief):
    def __init__(self,
                 robot_state,
                 target_objects,
                 search_region,
                 belief_config):
        self.robot_id = robot_state["id"]
        self.search_region = search_region
        self.target_objects = target_objects
        robot_belief = pomdp_py.Histogram({robot_state:1.0})
        prior = belief_config.get("prior", {})
        if type(prior) != dict:
            prior = {}

        object_beliefs = {robot_state["id"]: robot_belief}
        for objid in target_objects:
            belief_dist = {}
            object_prior_dist = prior.get(objid, {})
            if type(object_prior_dist) != dict:
                object_prior_dist = {}
            target = target_objects[objid]
            for loc in search_region:
                state = ObjectState2D(objid, target["class"], loc)
                if loc in object_prior_dist:
                    belief_dist[state] = object_prior_dist[loc]
                else:
                    # uniform
                    belief_dist[state] = 1.0 / len(search_region)
            object_beliefs[objid] = pomdp_py.Histogram(belief_dist)
        super().__init__(object_beliefs)


    def update_object_belief(self, agent, objid, zobj,
                             next_robot_state, action,
                             observation_model=None):
        """
        will update the belief with the given observation_model
        (default its the agent's built-in; But it may be set to the
        spatial language observation model, which is separate
        from the agent's model for planning.
        """
        assert isinstance(agent.transition_model.transition_models[objid],
                          StaticObjectTransitionModel)
        belief_dist = {}
        for loc in self.search_region:
            objclass = self.target_objects[objid]["class"]
            object_state = ObjectState2D(objid, objclass, loc)
            snext = pomdp_py.OOState({
                objid: object_state,
                self.robot_id: next_robot_state
            })
            if observation_model is None:
                observation_model = agent.observation_model[objid]
            pr_z = agent.observation_model[objid].probability(zobj, snext, action)
            belief_dist[object_state] = pr_z * self.b(objid)[object_state]

        belief_dist = pomdp_py.Histogram(normalize(belief_dist))
        self.set_object_belief(objid, belief_dist)

    def update_robot_belief(self, observation, action):
        # Note: assumes robot state observable
        next_robot_state = RobotState2D.from_obz(observation.z(self.robot_id))
        self.set_object_belief(
            self.robot_id, pomdp_py.Histogram({next_robot_state: 1.0}))
