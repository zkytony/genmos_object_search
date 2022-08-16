# utility functions for belief. The joint belief is modeled by
# pomdp_py.OOBelief
import pomdp_py

def init_object_belief(objid, objclass, search_region, object_prior=None):
    """prior: dictionary objid->{loc->prob}"""
    belief_dist = {}
    for loc in search_region:
        state = ObjectState(objid, objclass, loc)
        if object_prior == "uniform":
            # uniform
            belief_dist[state] = 1.0 / len(search_region)
        else:
            if loc in object_prior:
                belief_dist[state] = object_prior_dist[loc]
            else:
                # unspecified, still uniform
                belief_dist[state] = 1.0 / len(search_region)
    return pomdp_py.Histogram(belief_dist)

def init_object_beliefs(target_objects, search_region, prior=None):
    """prior: dictionary objid->{loc->prob}"""
    object_beliefs = {}
    if prior is None:
        prior = {}
    for objid in target_objects:
        target = target_objects[objid]
        object_prior = prior.get(objid, {})
        object_beliefs[objid] = Belief2D.init_object_belief(
            objid, target['class'], search_region, prior=object_prior)
    return object_beliefs
