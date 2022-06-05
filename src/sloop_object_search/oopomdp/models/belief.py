import pomdp_py

class BeliefBasic2D(pomdp_py.OOBelief):
    def __init__(self,
                 robot_state,
                 target_objects,
                 search_region,
                 belief_config):
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
