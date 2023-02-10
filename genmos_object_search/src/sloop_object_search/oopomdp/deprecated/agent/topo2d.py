"""
Agent that works over a topological graph
"""
import random
from collections import deque
import pomdp_py
from tqdm import tqdm
from sloop.agent import SloopAgent
from genmos_object_search.utils.osm import osm_map_to_grid_map
from genmos_object_search.utils.math import euclidean_dist, normalize
from genmos_object_search.oopomdp.deprecated.models.belief import Belief2D, BeliefTopo2D
from .basic2d import (init_detection_models,
                      init_object_transition_models)
from genmos_object_search.oopomdp.models.observation_model import (GMOSObservationModel,
                                                                  RobotObservationModelTopo)
from genmos_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from genmos_object_search.oopomdp.deprecated.models.policy_model import PolicyModelTopo
from genmos_object_search.oopomdp.deprecated.models.transition_model import RobotTransTopo
from genmos_object_search.oopomdp.deprecated.domain.state import RobotStateTopo
from genmos_object_search.oopomdp.deprecated.models.topo_map import TopoNode, TopoMap, TopoEdge


class SloopMosTopo2DAgent(SloopAgent):
    """
    topo --> operates at the topological graph level
    (Note that the topological graph is 2D)

    Note that the belief over target objects are still
    at the low-level.
    """
    def _init_oopomdp(self, grid_map=None):
        agent_config = self.agent_config
        if grid_map is None:
            # No grid map is provided. For now, we assume self.map_name is an OSM map
            self.grid_map = osm_map_to_grid_map(
                self.mapinfo, self.map_name)
        else:
            self.grid_map = grid_map

        # Prep work
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        target_ids = agent_config["targets"]
        self.target_objects = {objid: objects[objid] for objid in target_ids}
        search_region = self.grid_map.filter_by_label("search_region")
        no_look = agent_config.get("no_look", True)
        h_angle_res = agent_config["topo_trans_args"]["h_angle_res"]

        # generate topo map (and initialize robot state)
        combined_dist, init_object_beliefs =\
            self._compute_object_beliefs_and_combine(init=True)
        init_robot_state = self._update_topo_map(combined_dist, init=True)

        # transition models and observation models
        detection_models = init_detection_models(agent_config)
        robot_trans_model = RobotTransTopo(robot["id"], target_ids,
                                           self.topo_map, detection_models,
                                           h_angle_res=h_angle_res,
                                           no_look=no_look)
        transition_models = {robot["id"]: robot_trans_model,
                             **init_object_transition_models(agent_config)}
        transition_model = pomdp_py.OOTransitionModel(transition_models)

        robot_observation_model = RobotObservationModelTopo(robot['id'])
        observation_model = GMOSObservationModel(
            robot["id"], detection_models,
            robot_observation_model=robot_observation_model,
            no_look=no_look)

        # Policy Model
        target_ids = agent_config["targets"]
        policy_model = PolicyModelTopo(target_ids,
                                       robot_trans_model,
                                       no_look=no_look)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        init_belief = BeliefTopo2D(self.target_objects,
                                   robot_state=init_robot_state,
                                   belief_config=agent_config["belief"],
                                   object_beliefs=init_object_beliefs,
                                   search_region=search_region)

        return (init_belief,
                policy_model,
                transition_model,
                observation_model,
                reward_model)

    def sensor(self, objid):
        return self.observation_model.detection_models[objid].sensor

    def _compute_object_beliefs_and_combine(self, init=False):
        if init:
            # need to consider prior
            object_beliefs = {}
            search_region = self.grid_map.filter_by_label("search_region")
            for objid in self.target_objects:
                target = self.target_objects[objid]
                object_beliefs[objid] =\
                    Belief2D.init_object_belief(objid, target['class'], search_region,
                                                self.agent_config["belief"].get("prior", {}))
        else:
            # object beliefs just come from current belief
            object_beliefs = {objid: self.belief.b(objid) for objid in self.target_objects}

        search_region = self.grid_map.filter_by_label("search_region")
        combined_dist = BeliefTopo2D.combine_object_beliefs(
            search_region, object_beliefs)
        return combined_dist, object_beliefs


    def _update_topo_map(self, combined_dist, init=False):
        """combined_dist (dict): mapping from location to probability"""
        # TODO: this way of obtaining reachable positions may not be the best
        reachable_positions = self.grid_map.filter_by_label("reachable_for_topo")
        if len(reachable_positions) == 0:
            print("Warning: using search region as reachable positions.")
            reachable_positions = self.grid_map.filter_by_label("search_region")

        topo_map_args = self.agent_config["topo_map_args"]
        print("Sampling topological graph...")
        if init:
            robot = self.agent_config["robot"]
            robot_pose = robot["init_pose"]
            objects_found = tuple()
            camera_direction = None
        else:
            srobot_old = self.belief.b(self.robot_id).mpe()
            robot_pose = srobot_old.pose
            objects_found = srobot_old.objects_found
            camera_direction = srobot_old.camera_direction

        topo_map = _sample_topo_map(combined_dist,
                                    reachable_positions,
                                    topo_map_args.get("num_place_samples", 10),
                                    degree=topo_map_args.get("degree", (3,5)),
                                    sep=topo_map_args.get("sep", 4.0),
                                    rnd=random.Random(topo_map_args.get("seed", 1001)),
                                    robot_pos=robot_pose[:2])
        self.topo_map = topo_map
        if not init:
            self.policy_model.update(topo_map)
        topo_nid = topo_map.closest_node(*robot_pose[:2])
        robot_state = RobotStateTopo(self.robot_id,
                                     robot_pose,
                                     objects_found,
                                     camera_direction,
                                     topo_nid)

        if init:
            return robot_state
        else:
            self.belief.set_object_belief(
                self.robot_id, pomdp_py.Histogram({robot_state: 1.0}))

    def update_belief(self, observation, action):
        super().update_belief(observation, action)
        # sample new topological graph based on updated belief
        combined_dist, object_beliefs =\
            self._compute_object_beliefs_and_combine()
        if self._should_resample_topo_map(combined_dist):
            self._update_topo_map(combined_dist)

            # This is necessary because the action space changes due to new
            # topo graph; this shouldn't hurt, theoretically; It is necessary in order
            # to prevent replanning goals from the same, out-dated tree while
            # a goal is in execution.
            if hasattr(self, "tree"):
                self.tree = None # remove the search tree after planning

    def _should_resample_topo_map(self, combined_dist):
        # We will compute coverage of the probabilities
        search_region = self.grid_map.filter_by_label("search_region")
        topo_map_args = self.agent_config["topo_map_args"]
        node_coverage_radius = topo_map_args.get("node_coverage_radius", 3.0)
        topo_nodes = list(self.topo_map.nodes.keys())
        topo_node_prob = {}  # nid to prob
        for loc in search_region:
            # find closest node
            closest_nid = min(
                topo_nodes, key=lambda nid: euclidean_dist(self.topo_map.nodes[nid].pos, loc))
            # If the distance is not too far away
            if euclidean_dist(self.topo_map.nodes[closest_nid].pos, loc) < node_coverage_radius:
                topo_node_prob[closest_nid]\
                    = topo_node_prob.get(closest_nid, 0.0) + combined_dist[loc]

        # combine the topo node probs --> the probability "covered" by the nodes.
        total_prob = sum(topo_node_prob[n] for n in topo_node_prob)
        print(f"Topo map coverage total prob: {total_prob}")
        return total_prob < topo_map_args.get("resample_prob_thres", 0.4)


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
    """Note: originally from COSPOMDP codebase; But
    modified - instead of creating a mapping from reachable position
    to a set of closest search region locations, simply sample from
    the distribution based on the search region locations, and then
    add a closest reachable position as a node. The separation and
    degree requirements still apply.

    Args:
        target_hist (dict): maps from search region location location to probability
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

    hist = pomdp_py.Histogram(normalize(target_hist))

    places = set()
    if robot_pos is not None:
        places.add(robot_pos)
    for i in tqdm(range(num_samples)):
        search_region_loc = hist.random(rnd=rnd)
        closest_reachable_pos = min(reachable_positions,
                                    key=lambda robot_pos: euclidean_dist(
                                        search_region_loc, robot_pos))
        pos = closest_reachable_pos
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
        topo_node = TopoNode(i, pos)
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
