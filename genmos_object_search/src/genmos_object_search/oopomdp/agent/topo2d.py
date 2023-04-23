"""This represents the global, 2D search agent that plans
over an action space based on a topological graph."""
import logging
import pomdp_py
import random
import time
from collections import deque
from tqdm import tqdm
from ..domain.state import RobotStateTopo
from ..domain.observation import RobotObservation, RobotObservationTopo
from ..models.policy_model import PolicyModelTopo
from ..models.transition_model import RobotTransTopo2D
from ..models.observation_model import RobotObservationModelTopo
from ..models.topo_map import TopoNode, TopoMap, TopoEdge
from ..models import belief
from .common import (MosAgent, init_object_transition_models,
                     interpret_localization_model, init_visualizer2d)
from .basic2d import MosAgentBasic2D
from genmos_object_search.utils import math as math_utils
from genmos_object_search.utils.algo import PriorityQueue
from genmos_object_search.utils import grid_map_utils


class MosAgentTopo2D(MosAgentBasic2D):
    """This agent will have a topological graph-based action space."""

    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None, **args_init_object_beliefs):
        # first, initialize object beliefs
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                belief_config=self.agent_config["belief"],
                **args_init_object_beliefs)

        # now, generate topological map
        mpe_robot_pose = tuple(init_robot_pose_dist.mean)
        self.topo_map = self.generate_topo_map(
            init_object_beliefs, mpe_robot_pose)

        # now, generate initial robot belief
        init_topo_nid = self.topo_map.closest_node(mpe_robot_pose[:2])
        init_robot_belief = belief.init_robot_belief(
            self.agent_config["robot"], init_robot_pose_dist,
            robot_state_class=RobotStateTopo,
            topo_nid=init_topo_nid,
            topo_map_hashcode=self.topo_map.hashcode)
        init_belief = pomdp_py.OOBelief({self.robot_id: init_robot_belief,
                                         **init_object_beliefs})
        return init_belief

    def init_robot_observation_model(self, localization_model):
        robot_observation_model = RobotObservationModelTopo(
            self.robot_id, localization_model=localization_model)
        return robot_observation_model

    def init_transition_and_policy_models(self):
        target_ids = self.agent_config["targets"]
        trans_args = self.agent_config["robot"].get("transition", {})
        robot_trans_model = RobotTransTopo2D(self.robot_id, target_ids,
                                             self.topo_map, self.detection_models,
                                             no_look=self.no_look)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        policy_model = PolicyModelTopo(target_ids,
                                       robot_trans_model,
                                       no_look=self.no_look)
        return transition_model, policy_model

    def _inflate_obstacles(self):
        inflated_cells = None
        if not hasattr(self, "obstacles2d"):
            grid_map = self.search_region.grid_map
            # we want to inflate the obstacles
            inflation = int(round(self.topo_config.get("inflation", 1)))
            shrunk_free_cells = grid_map_utils.cells_with_minimum_distance_from_obstacles(grid_map, dist=inflation)
            inflated_cells = (grid_map.free_locations - shrunk_free_cells)
            self.obstacles2d = grid_map.obstacles | inflated_cells | grid_map.unknowns
        return inflated_cells

    def generate_topo_map(self, object_beliefs, robot_pose):
        """Given 'combined_dist', a distribution that maps
        location to the sum of prob over all objects, and
        the current 'robot_pose', sample a topological graph
        based on this distribution.
        """
        inflated_cells = self._inflate_obstacles()  # affects 'reachable' function

        print("Sampling topological graph...")
        topo_map = _sample_topo_map(object_beliefs,
                                    robot_pose,
                                    self.search_region,
                                    self.reachable,
                                    self.topo_config)
        _debug = self.topo_config.get("debug", False)
        if _debug:
            try:
                from genmos_object_search.utils import visual2d
                viz = init_visualizer2d(visual2d.VizSloopMosTopo, self.agent_config,
                                        grid_map=self.search_region.grid_map,
                                        res=self.visual_config.get("res", 10))
                img = viz.render(topo_map, object_beliefs,
                                 self.robot_id, robot_pose)
                if inflated_cells is not None:
                    img = viz.highlight(img, inflated_cells, color=(72, 213, 235))
                # flip horizotnally is necessary so that +x is right, +y is up.
                viz.show_img(img, flip_horizontally=True)
                time.sleep(2)
                viz.on_cleanup()
            except ImportError as ex:
                logging.error("Error importing visual2d: {}".format(ex))

        return topo_map

    def reachable(self, pos):
        return pos not in self.obstacles2d

    @property
    def topo_config(self):
        return self.agent_config["robot"]["action"].get("topo", {})

    def _update_robot_belief(self, observation, action=None, **kwargs):
        """from the perspective of the topo2d agent, it just needs to care about updating
        its own belief, which is 2D"""
        # obtain topo node; we only update topo node when action finishes.
        if action is not None:
            # this means action execution has finished
            topo_nid = self.topo_map.closest_node(observation.loc)
            topo_map_hashcode = self.topo_map.hashcode
        else:
            current_srobot_mpe = self.belief.mpe().s(self.robot_id)
            topo_nid = current_srobot_mpe.topo_nid
            topo_map_hashcode = current_srobot_mpe.topo_map_hashcode
        super()._update_robot_belief(observation, action=action,
                                     robot_state_class=RobotStateTopo,
                                     topo_nid=topo_nid,
                                     topo_map_hashcode=topo_map_hashcode)

    def update_belief(self, observation, action=None, debug=False, **kwargs):
        _aux = super().update_belief(observation, action=action, debug=debug, **kwargs)
        if isinstance(observation, RobotObservation):
            # The observation doesn't lead to object belief change. We are done.
            return _aux

        robot_observation = observation.z(self.robot_id)

        # Check if we need to resample topo map - check with objects not yet found
        object_beliefs = {objid: self.belief.b(objid)
                          for objid in self.belief.object_beliefs
                          if objid != self.robot_id\
                          and objid not in robot_observation.objects_found}
        if len(object_beliefs) > 0:
            # there exist unfound objects
            if self.should_resample_topo_map(object_beliefs):
                robot_pose = robot_observation.pose
                topo_map = self.generate_topo_map(
                    object_beliefs, robot_pose)
                self._update_topo_map(topo_map, robot_observation, action=action)
        return _aux

    def _update_topo_map(self, topo_map, robot_observation, action=None):
        """
        we expect robot_observation to be RobotObservation which
        contains a RobotLocalization; This is checked by update_robot_belief
        """
        self.topo_map = topo_map
        self.policy_model.update(topo_map)
        # Now, we need to update the robot belief because new topo map leads
        # to new topo nid and hash
        topo_nid = self.topo_map.closest_node(robot_observation.loc)
        super()._update_robot_belief(robot_observation, action=action,
                                     robot_state_class=RobotStateTopo,
                                     topo_nid=topo_nid,
                                     topo_map_hashcode=self.topo_map.hashcode)

    def should_resample_topo_map(self, object_beliefs):
        zone_res = self.topo_config.get("zone_res", 8)
        resample_prob_thres = self.topo_config.get("resample_thres", 0.3)
        total_prob = 0
        zones_covered = set()  # set of zones whose area's probability has been considered
        for nid in self.topo_map.nodes:
            pos = self.topo_map.nodes[nid].pos
            zone_pos = (pos[0] // zone_res,
                        pos[1] // zone_res)
            if zone_pos in zones_covered:
                continue
            prob = _compute_combined_prob_around(pos, object_beliefs, zone_res)
            total_prob += prob
            zones_covered.add(zone_pos)
        # total_prob should be a normalized probability
        if not 0 <= total_prob <= 1:
            print(f"invalid total prob:  {total_prob}")
        print("total prob covered by existing topo map nodes:", total_prob)
        return total_prob < resample_prob_thres


def _compute_combined_prob_around(pos, object_beliefs, zone_res=8):
    """Given a position 'pos', and object beliefs, and a resolution level for the
    zone which the belief around the position is considered, return a normalized
    probability that combines the beliefs over all objects within that zone.
    """
    comb_prob = 0
    for objid in object_beliefs:
        if not isinstance(object_beliefs[objid], belief.ObjectBelief2D):
            raise ValueError("topo graph3d requires object beliefs to be OctreeBelief")
        b_obj = object_beliefs[objid]
        comb_prob += b_obj.loc_dist.prob_in_rect(pos, zone_res, zone_res)
    # because 'prob_at' returns normalized prob for each object, the normalizer
    # of the combinations is just the number of objects.
    return comb_prob / len(object_beliefs)


def _sample_topo_map(init_object_beliefs,
                     init_robot_pose,
                     search_region,
                     reachable_func,
                     topo_config={}):
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
    num_nodes = topo_config.get("num_nodes", 10)
    num_samples = topo_config.get("num_samples", 1000)
    degree = topo_config.get("degree", (3,5))
    sep = topo_config.get("sep", 4.0)
    rnd = random.Random(topo_config.get("seed", 1000))
    zone_res = topo_config.get("zone_res", 8)
    pos_importance_thres = topo_config.get("pos_importance_thres", 0.3)

    if type(degree) == int:
        degree_range = (degree, degree)
    else:
        degree_range = degree
        if len(degree_range) != 2:
            raise ValueError("Invalid argument for degree {}."
                             "Accepts int or (int, int)".format(degree))

    if isinstance(init_object_beliefs, pomdp_py.OOBelief):
        init_object_beliefs = init_object_beliefs.object_beliefs

    # The overall idea: sample robot positions from within the search region,
    # and rank them based on object beliefs, and only keep <= X number of nodes
    # that have normalized scores above some threshold
    grid_map = search_region.grid_map  # should be GridMap2
    candidate_positions = set()
    if reachable_func(init_robot_pose[:2]):
        candidate_positions.add(init_robot_pose[:2])
    candidate_scores = []
    min_prob = float("inf")
    max_prob = float("-inf")
    for i in range(num_samples):
        shifted_x = rnd.uniform(0, grid_map.width-1)
        shifted_y = rnd.uniform(0, grid_map.length-1)
        x, y = grid_map.shift_back_pos(shifted_x, shifted_y)
        pos = (int(round(x)), int(round(y)))
        added = False
        if reachable_func(pos):
            if len(candidate_positions) == 0:
                candidate_positions.add(pos)
                added = True
            else:
                closest = min(candidate_positions,
                              key=lambda c: math_utils.euclidean_dist(pos, c))
                if math_utils.euclidean_dist(closest, pos) >= sep:
                    candidate_positions.add(pos)
                    added = True

        if added:
            prob_pos =\
                _compute_combined_prob_around(pos, init_object_beliefs, zone_res=zone_res)
            candidate_scores.append((pos, prob_pos))
            min_prob = min(min_prob, prob_pos)
            max_prob = max(max_prob, prob_pos)

    pq = PriorityQueue()
    positions = []
    if reachable_func(init_robot_pose[:2]):
        positions.append(init_robot_pose[:2])
    for pos, prob_pos in candidate_scores:
        norm_score = (prob_pos - min_prob) / (max_prob - min_prob)
        if norm_score > pos_importance_thres:
            pq.push(pos, -norm_score)
    while not pq.isEmpty() and len(positions) < num_nodes:
        positions.append(pq.pop())

    # Create nodes
    pos_to_nid = {}
    nodes = {}
    for i, pos in enumerate(positions):
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
        candidates = set(positions) - {nodes[nid].pos} - neighbor_positions
        degree_needed = degree_range[0] - len(neighbors)
        if degree_needed <= 0:
            continue
        new_neighbors = list(sorted(candidates, key=lambda pos: math_utils.euclidean_dist(pos, nodes[nid].pos)))[:degree_needed]
        for nbpos in new_neighbors:
            nbnid = pos_to_nid[nbpos]
            if nbnid not in _conns or len(_conns[nbnid]) < degree_range[1]:
                _conns[nid].add(nbnid)
                if nbnid not in _conns:
                    _conns[nbnid] = set()
                _conns[nbnid].add(nid)

                #TODO: treating free locations as reachable
                path = _shortest_path(search_region.grid_map.free_locations,
                                      nodes[nbnid].pos,
                                      nodes[nid].pos)
                if path is None:
                    # Skip this edge because we cannot find path
                    continue
                eid = len(edges) + 1000
                edges[eid] = TopoEdge(eid,
                                      nodes[nid],
                                      nodes[nbnid],
                                      {"path": path, "length": len(path)})
    if len(edges) == 0:
        edges[0] = TopoEdge(0, nodes[next(iter(nodes))], None, [])

    topo_map = TopoMap(edges)
    # Verification
    for nid in topo_map.nodes:
        assert len(topo_map.edges_from(nid)) <= degree_range[1]

    return topo_map

def _shortest_path(reachable_positions, gloc1, gloc2):
    """
    Note: originally from COSPOMDP codebase

    Computes the shortest distance between two locations.
    The two locations will be snapped to the closest free cell.

    TODO: This is limited to 2D. The topo map sampling process
    should be generalized to beyond 2D grid maps.
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


### DEPRECATED ###
try:
    from .common import SloopMosAgent
    class SloopMosAgentTopo2D(SloopMosAgent):
        def _init_oopomdp(self, init_robot_pose_dist=None, init_object_beliefs=None):
            if init_robot_pose_dist is None:
                raise ValueError("To instantiate MosAgent, initial robot pose distribution is required.")

            mos_agent = MosAgentTopo2D(self.agent_config,
                                        self.search_region,
                                        init_robot_pose_dist=init_robot_pose_dist,
                                        init_object_beliefs=init_object_beliefs)
            return (mos_agent.belief,
                    mos_agent.policy_model,
                    mos_agent.transition_model,
                    mos_agent.observation_model,
                    mos_agent.reward_model)

except ImportError as ex:
    logging.error("Failed to import SloopMosAgent (basic2d): {}".format(ex))
