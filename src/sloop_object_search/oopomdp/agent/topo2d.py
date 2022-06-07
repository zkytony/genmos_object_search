"""
Agent that works over a topological graph
"""
import pomdp_py
from sloop.agent import SloopAgent
from sloop_object_search.utils.osm import osm_map_to_grid_map
from ..models.belief import Belief2D, BeliefTopo2D
from .basic2d import (init_detection_models,
                      init_object_transition_models)

class SloopMosTopo2DAgent(SloopAgent):
    """
    topo --> operates at the topological graph level
    (Note that the topological graph is 2D)

    Note that the belief over target objects are still
    at the low-level.
    """
    def _init_oopomdp(self):
        agent_config = self.agent_config
        self.grid_map = osm_map_to_grid_map(
            self.mapinfo, self.map_name)

        # Prep work
        objects = agent_config["objects"]
        target_ids = agent_config["targets"]
        target_objects = {objid: objects[objid]
                          for objid in target_ids}
        search_region = self.grid_map.filter_by_label("search_region")

        # initial object beliefs and combine them together
        init_object_beliefs = {}
        for objid in target_ids:
            init_object_beliefs[objid] =\
                Belief2D.init_object_belief(objid, search_region,
                                            agent_config["belief"].get("prior", {}))
        combined_dist = BeliefTopo2D.combine_object_beliefs(search_region,
                                                            init_object_beliefs)
        reachable_positions = self.grid_map.filter_by_label("reachable")
        _topo_map_args = agent_config["topo_map_args"]
        self.topo_map = _sample_topo_map(combined_dist,
                                         reachable_positions,
                                         _topo_map_args.get("num_place_samples", 10),
                                         degree=_topo_map_args.get("degree", (3,5)),
                                         sep=_topo_map_args.get("sep", 4.0),
                                         rnd=random.Random(_topo_map_args.get("seed", 1001))
        )


        init_topo_nid = self.topo_map.closest_node(*robot["init_pose"][:2])
        init_robot_state = RobotStateTopo(robot["id"],
                                          robot["init_pose"],
                                          robot.get("found_objects", tuple()),
                                          robot.get("camera_direction", None),
                                          init_topo_nid)
        # transition models and observation models
        detection_models = init_detection_models(agent_config)
        robot_trans_model = RobotTransTopo(robot["id"], self.topo_map, detection_models)
        transition_models = {**{robot["id"]: robot_trans_model},
                             **init_object_transition_models(agent_config)}

        robot_observation_model = RobotObservationModelTopo(robot['id'])
        observation_model = GMOSObservationModel(
            robot["id"], detection_models,
            robot_observation_model=robot_observation_model,
            no_look=no_look)

        # Policy Model
        target_ids = agent_config["targets"]
        policy_model = PolicyModelTopo(target_ids,
                                       robot_trans_model,
                                       action_scheme,
                                       observation_model,
                                       no_look=no_look)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        init_belief = BeliefTopo2D(init_robot_state,
                                   target_objects,
                                   search_region,
                                   agent_config["belief"],
                                   object_beliefs=init_object_beliefs)

        return (init_belief,
                policy_model,
                transition_model,
                observation_model,
                reward_model)




def _shortest_path(reachable_positions, gloc1, gloc2):
    """
    Note: originally from COSPOMDP codebase

    Computes the shortest distance between two locations.
    The two locations will be snapped to the closest free cell.
    """
    def neighbors(x,y):
        return [(x+1, y), (x-1,y),
                (x,y+1), (x,y-1)]

    def get_path(s, t, prev):
        v = t
        path = [t]
        while v != s:
            v = prev[v]
            path.append(v)
        return path

    # BFS; because no edge weight
    reachable_positions = set(reachable_positions)
    visited = set()
    q = deque()
    q.append(gloc1)
    prev = {gloc1:None}
    while len(q) > 0:
        loc = q.popleft()
        if loc == gloc2:
            return get_path(gloc1, gloc2, prev)
        for nb_loc in neighbors(*loc):
            if nb_loc in reachable_positions:
                if nb_loc not in visited:
                    q.append(nb_loc)
                    visited.add(nb_loc)
                    prev[nb_loc] = loc
    return None


def _sample_topo_map(target_hist,
                     reachable_positions,
                     num_samples,
                     degree=(3,5),
                     sep=4.0,
                     rnd=random,
                     robot_pos=None):
    """Note: originally from COSPOMDP codebase

    Given a search region, a distribution over target locations in the
    search region, return a TopoMap with nodes within
    reachable_positions.

    The algorithm works by first converting the target_hist,
    which is a distribution over the search region, to a distribution
    over the robot's reachable positions.

    This is done by, for each location in the search region, find
    a closest reachable position; Then the probability at a reachable
    position is the sum of those search region locations mapped to it.

    Then, simply sample reachable positions based on this distribution.

    The purpose of this topo map is for navigation action abstraction
    and robot state abstraction.

    Args:
        target_hist (dict): maps from location to probability
        reachable_positions (list of tuples)
        num_places (int): number of places to sample
        degree (int or tuple): Controls the minimum and maximum degree
            per topo node in the resulting graph. If only one number is
            passed, then will make all nodes have the same degree.
            This assumes there are enough sampled nodes to satisfy this
            requirement; If not, then all nodes are still guaranteed
            to have degree less than or equal to the maximum degree.
        sep (float): minimum distance between two places (grid cells)
        robot_pos (x,y): If not None, will add a node at where the robot is.

    Returns:
        TopologicalMap.

    """
    if type(degree) == int:
        degree_range = (degree, degree)
    else:
        degree_range = degree
        if len(degree_range) != 2:
            raise ValueError("Invalid argument for degree {}."
                             "Accepts int or (int, int)".format(degree))

    mapping = {}  # maps from reachable pos to a list of search region locs
    for loc in target_hist:
        closest_reachable_pos = min(reachable_positions,
                                    key=lambda robot_pos: euclidean_dist(loc, robot_pos))
        if closest_reachable_pos not in mapping:
            mapping[closest_reachable_pos] = []
        mapping[closest_reachable_pos].append(loc)

    # distribution over reachable positions
    reachable_pos_dist = {}
    for pos in mapping:
        reachable_pos_dist[pos] = 0.0
        for search_region_loc in mapping[pos]:
            reachable_pos_dist[pos] += target_hist[search_region_loc]
    hist = pomdp_py.Histogram(normalize(reachable_pos_dist))

    places = set()
    if robot_pos is not None:
        places.add(robot_pos)
    for i in range(num_samples):
        pos = hist.random(rnd=rnd)
        if len(places) > 0:
            closest_pos = min(places,
                              key=lambda c: euclidean_dist(pos, c))
            if euclidean_dist(closest_pos, pos) >= sep:
                places.add(pos)
        else:
            places.add(pos)

    # Create nodes
    pos_to_nid = {}
    nodes = {}
    for i, pos in enumerate(places):
        topo_node = TopoNode(i, pos, mapping.get(pos, set()))
        nodes[i] = topo_node
        pos_to_nid[pos] = i

    # Now, we need to connect the places to form a graph.
    _conns = {}
    edges = {}
    for nid in nodes:
        if nid not in _conns:
            _conns[nid] = set()
        neighbors = _conns[nid]
        neighbor_positions = {nodes[nbnid].pos for nbnid in neighbors}
        candidates = set(places) - {nodes[nid].pos} - neighbor_positions
        degree_needed = degree_range[0] - len(neighbors)
        if degree_needed <= 0:
            continue
        new_neighbors = list(sorted(candidates, key=lambda pos: euclidean_dist(pos, nodes[nid].pos)))[:degree_needed]
        for nbpos in new_neighbors:
            nbnid = pos_to_nid[nbpos]
            if nbnid not in _conns or len(_conns[nbnid]) < degree_range[1]:
                _conns[nid].add(nbnid)
                if nbnid not in _conns:
                    _conns[nbnid] = set()
                _conns[nbnid].add(nid)

                path = _shortest_path(reachable_positions,
                                      nodes[nbnid].pos,
                                      nodes[nid].pos)
                if path is None:
                    # Skip this edge because we cannot find path
                    continue
                eid = len(edges) + 1000
                edges[eid] = TopoEdge(eid,
                                      nodes[nid],
                                      nodes[nbnid],
                                      path)
    if len(edges) == 0:
        edges[0] = TopoEdge(0, nodes[next(iter(nodes))], None, [])

    topo_map = TopoMap(edges)
    # Verification
    for nid in topo_map.nodes:
        assert len(topo_map.edges_from(nid)) <= degree_range[1]

    return topo_map
