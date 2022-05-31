# Contains functions that converts a language
# to an observation, and use that observation
# to update agent belief.

import pomdp_py
from .problem import *
from ..parsing.parser import *
from ..parsing.graph.ego import *
from ..parsing.graph.spatial import *
from ..parsing.graph import util


def splang_to_spgraph(lang):
    """
    Input: lang (str)
    Output: spatial_graph
    """
    raise NotImplementedError("This is the NLP problem Language->SpatialGraph")


def spgraph_to_egos(sg, grid_map): # result of parsing the language
    """
    Input: sg (SpatialGraph)
           grid_map (GridMap)
    Output: ego_graphs (list), a list of EgoGraphs about the landmarks
                in the sptial graph where the nodes contain likelihoods.
    """
    raise NotImplementedError("This is the NLP problem Language->SpatialGraph")


def egos_to_obs(objects,   # list of object ids
                grid_map,  # grid map
                eg_list,
                specs=[],
                default_dist=1):
    """
    Input: objects (list) List of object ids.
           grid_map (GridMap) the occupancy grid map
           eg_list (list), a list of EgoGraphs about the landmarks
                in the sptial graph where the nodes contain likelihoods.
           grid_map (GridMap)
           specs (list): that specifies the grounding parameters (i.e.
               distances to the center from view nodes, phase shifts, for now).
               If a distance is not provided, it is assumed to be default dist.
    Output: a dictionary, mapping object id to
                a matrix of beliefs at different grid cells.
    """
    # ground the ego graphs to produce grounded ego graphs
    geg_list = []
    for i, eg in enumerate(eg_list):
        if i < len(specs):
            assert -1 in specs[i],\
                "The location of center must be specified for %s" % eg.name
            for viewnum in range(eg.divisions):
                if viewnum not in specs[i]:
                    specs[i][viewnum] = default_dist
        geg = GroundedEgoGraph.ground(eg, specs[i], grid_map)
        geg_list.append(geg)

    # Go from geg to belief; Iterate over every grid cell,
    # compute a score based on distance to closest view nodes
    # of each landmark. Normalize over that score to produce
    # probability distribution (as belief).
    oo_spl_obsrv = {}
    for objid in objects:
        spl_obsrv = {}  # spatial language observation, a map from location to belief
        total_score = 0
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                score = compute_score(x, y, objid, geg_list)
                spl_obsrv[(x,y)] = score
                total_score += score
        # normalize
        for loc in spl_obsrv:
            spl_obsrv[loc] /= total_score
        oo_spl_obsrv[objid] = spl_obsrv
    return oo_spl_obsrv, geg_list


def compute_score(x, y, objid, geg_list, dist_factor=0.1):
    # For each grounded ego graph, find the
    # node closest to the location (x,y).
    # The score is simply the sum of those likelihoods,
    # for now.
    score = 0
    for geg in geg_list:
        closest_viewnum = min([viewnum
                               for viewnum in range(geg.divisions)],
                              key=lambda viewnum: util.euclidean_dist(
                                  (x,y), geg.coords_by_viewnum(viewnum)))
        lh = geg.node_at(closest_viewnum).lh_obj(objid)
        score += lh / (dist_factor * util.euclidean_dist(
            (x,y), geg.coords_by_viewnum(closest_viewnum)))
    return score


def splang_matrix_belief_update(oo_spl_obsrv, agent):
    """Updates agent belief as a matrix operation"""
    assert isinstance(agent.belief, pomdp_py.OOBelief),\
        "Agent should have OOBelief."
    for objid in agent.belief.object_beliefs:
        if objid == agent.robot_id:
            continue
        belief_obj = agent.belief.object_beliefs[objid]
        assert isinstance(belief_obj, pomdp_py.Histogram),\
            "Expecting object belief to be a histogram"
        new_histogram = {}
        total_prob = 0
        for state in belief_obj:
            lh_obs = oo_spl_obsrv[objid][state.pose]
            new_histogram[state] = belief_obj[state] * lh_obs
            total_prob += new_histogram[state]
        # normalize
        for state in new_histogram:
            new_histogram[state] /= total_prob
        new_belief_obj = pomdp_py.Histogram(new_histogram)
        agent.belief.set_object_belief(objid, new_belief_obj)
