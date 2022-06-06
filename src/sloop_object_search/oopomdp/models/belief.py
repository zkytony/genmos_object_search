import pomdp_py
from .transition_model import StaticObjectTransitionModel
from .observation_model import robot_state_from_obz
from ..domain.state import ObjectState
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
        if prior == "uniform":
            prior = {}

        object_beliefs = {robot_state["id"]: robot_belief}
        for objid in target_objects:
            belief_dist = {}
            object_prior_dist = prior.get(objid, {})
            target = target_objects[objid]
            for loc in search_region:
                state = ObjectState(objid, target["class"], loc)
                if loc in object_prior_dist:
                    belief_dist[state] = object_prior_dist[loc]
                else:
                    # uniform
                    belief_dist[state] = 1.0 / len(search_region)
            object_beliefs = {objid: pomdp_py.Histogram(belief_dist)}
        super().__init__(object_beliefs)

    def update_object_belief(self, objid, zobj, action, agent,
                             observation_model=None):
        belief_dist = {}
        for loc in self.search_region:
            objclass = self.target_objects[objid]["class"]
            object_state = ObjectState(objid, objclass, loc)
            zobj = observation.z(objid)
            snext = pomdp_py.OOState({
                objid: object_state,
                self.robot_id: next_robot_state
            })
            if observation_model is None:
                observation_model = agent.observation_model[objid]
            pr_z = agent.observation_model[objid].probability(zobj, snext, action)
            belief_dist[object_state] = pr_z * self.b(objid)[object_state]

        belief_dist = normalize(belief_dist)
        self.set_object_belief(objid, belief_dist)
