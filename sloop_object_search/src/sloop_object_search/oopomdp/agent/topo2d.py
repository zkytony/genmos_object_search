"""This represents the global, 2D search agent that plans
over an action space based on a topological graph."""
import pomdp_py
import random
import time
from collections import deque
from tqdm import tqdm
from sloop_object_search.utils.math import euclidean_dist, normalize
from sloop_object_search.utils import visual2d
from ..domain.state import RobotStateTopo
from ..domain.observation import RobotObservation, RobotObservationTopo
from ..models.policy_model import PolicyModelTopo
from ..models.transition_model import RobotTransTopo2D
from ..models.observation_model import RobotObservationModelTopo
from ..models.topo_map import TopoNode, TopoMap, TopoEdge
from .common import (MosAgent, SloopMosAgent, init_object_transition_models,
                     interpret_localization_model, init_visualizer2d)
from . import belief

class MosAgentTopo2D(MosAgent):
    """This agent will have a topological graph-based action space."""

    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None):
        # first, initialize object beliefs
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                belief_config=self.agent_config["belief"])

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
        h_angle_res = trans_args.get("h_angle_res", 45.0)
        robot_trans_model = RobotTransTopo2D(self.robot_id, target_ids,
                                             self.topo_map, self.detection_models,
                                             h_angle_res=h_angle_res,
                                             no_look=self.no_look)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        policy_model = PolicyModelTopo(target_ids,
                                       robot_trans_model,
                                       no_look=self.no_look)
        return transition_model, policy_model

    def generate_topo_map(self, object_beliefs, robot_pose, debug=True):
        """Given 'combined_dist', a distribution that maps
        location to the sum of prob over all objects, and
        the current 'robot_pose', sample a topological graph
        based on this distribution.
        """
        combined_dist = belief.accumulate_object_beliefs(
            self.search_region, object_beliefs)
        position_candidates = self.search_region.grid_map.free_locations
        if len(position_candidates) == 0:
            raise ValueError("No position candidates for topo map sampling.")
        topo_map_args = self.agent_config.get("topo_map_args", {})
        print("Sampling topological graph...")
        topo_map = _sample_topo_map(combined_dist,
                                    position_candidates,
                                    topo_map_args.get("num_place_samples", 10),
                                    degree=topo_map_args.get("degree", (3,5)),
                                    sep=topo_map_args.get("sep", 4.0),
                                    rnd=random.Random(topo_map_args.get("seed", 1001)),
                                    robot_pos=robot_pose[:2])
        if debug:
            viz = init_visualizer2d(visual2d.VizSloopMosTopo, self.agent_config,
                                    grid_map=self.search_region.grid_map,
                                    res=self.visual_config.get("res", 10))
            img = viz.render(topo_map, object_beliefs,
                             self.robot_id, robot_pose)
            # flip horizotnally is necessary so that +x is right, +y is up.
            viz.show_img(img, flip_horizontally=True)
            time.sleep(2)
            viz.on_cleanup()
        return topo_map

    def _update_robot_belief(self, observation, action=None, **kwargs):
        """from the perspective of the topo2d agent, it just needs to care about updating
        its own belief, which is 2D"""
        # obtain topo node
        current_srobot_mpe = self.belief.mpe().s(self.robot_id)
        super()._update_robot_belief(observation_2d, action=action,
                                     robot_state_class=RobotStateTopo,
                                     topo_nid=current_srobot_mpe.topo_nid,
                                     topo_map_hashcode=current_srobot_mpe.topo_map_hashcode)

    def _update_object_beliefs(self, observation, action=None, debug=False, **kwargs):
        """the object detections and robot obseravtion in 'observation' should be 2D; we
        need to update the 2D belief given this 3D observation.

        There are two kinds of situations for belief update for the 2D topo agent.
        (1) Local search is happening. Then, the belief of this agent will be updated
            according to the local agent's belief.
        (2) Local search is not happening. Then, the belief of this agent will be
            updated based on the observation. Note that we assume the robot does not
            tilt its camera actively during 2D search - that means, the 2D sensor model
            appropriately approximates the FOV during 2D search.
        """
        searching_locally = kwargs.pop("searching_locally", False)
        if not searching_locally:
            # convert all 3D observations to 2D.
            # First, do the robot observation conversion
            zrobot_3d = observation.z(self.robot_id)
            zrobot_2d = RobotObservation(zrobot_3d.robot_id,
                                         zrobot_3d.pose_est.to_2d(),
                                         zrobot_3d.objects_found,
                                         zrobot_3d.camera_direction)
            for objid in observation:
                zobj = observation.z(objid)

        import pdb; pdb.set_trace()



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
