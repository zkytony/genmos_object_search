# utility functions for belief. The joint belief is modeled by
# pomdp_py.OOBelief
import pomdp_py
from ..models.search_region import SearchRegion2D, SearchRegion3D
from ..domain.state import ObjectState

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
