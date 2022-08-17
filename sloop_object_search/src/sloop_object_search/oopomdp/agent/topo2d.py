import pomdp_py
from ..domain.state import RobotStateTopo
from .agent import MosAgent
from . import belief

class MosAgentTopo2D(MosAgent):
    """This agent will have a topological graph-based action space."""

    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None):
        # first, initialize object beliefs
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                prior=self.agent_config["belief"].get("prior", {}))

        # now, generate topological map
        combined_dist = belief.accumulate_object_beliefs(
            self.search_region, self.init_object_beliefs)
        mpe_robot_pose = init_robot_pose_dist.mpe()
        self.topo_map = self.generate_topo_map(combined_dist, mpe_robot_pose)

        # now, generate initial robot belief
        init_topo_nid = topo_map.closest_node(*mpe_robot_pose[:2])
        init_robot_belief = belief.init_robot_belief(
            self.agent_config["robot"], init_robot_pose_dist,
            robot_state_class=RobotStateTopo,
            topo_nid=init_topo_nid)
        init_belief = pomdp_py.OOBelief({self.robot_id: init_robot_belief,
                                         **init_object_beliefs})
        return init_belief

    def init_transition_and_policy_models(self):
        # we need to first sample a topo map.
        combined_dist = combine_object_beliefs(
            self.search_region, self.init_object_beliefs)
        self._update_topo_map(combined_dist, init=True)

        trans_args = self.agent_config["robot"].get("transition", {})
        h_angle_res = trans_args.get("h_angle_res", 45.0)
        robot_trans_model = RobotTransTopo(robot["id"], target_ids,
                                           self.topo_map, self.detection_models,
                                           h_angle_res=h_angle_res,
                                           no_look=no_look)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

    def generate_topo_map(self, combined_dist, robot_pose):
        """Given 'combined_dist', a distribution that maps
        location to the sum of prob over all objects, and
        the current 'robot_pose', sample a topological graph
        based on this distribution.
        """
        # TODO: We should avoid explicit list of reachable positions!
        # (This involves improvement of the topo map sampling algorithm)
        reachable_positions = self.search_region.grid_map.free_locations
        if len(reachable_positions) == 0:
            print("Warning: using search region as reachable positions.")
            reachable_positions = self.grid_map.filter_by_label("search_region")
        topo_map_args = self.agent_config.get("topo_map_args", {})
        print("Sampling topological graph...")
        topo_map = _sample_topo_map(combined_dist,
                                    reachable_positions,
                                    topo_map_args.get("num_place_samples", 10),
                                    degree=topo_map_args.get("degree", (3,5)),
                                    sep=topo_map_args.get("sep", 4.0),
                                    rnd=random.Random(topo_map_args.get("seed", 1001)),
                                    robot_pos=robot_pose[:2])
        return topo_map


    # def _update_topo_map(self, combined_dist, robot_pose):
    #     """combined_dist (dict): mapping from location to probability"""
    #     if init:
    #         robot_pose = self.init_robot_belief.mpe().pose
    #         objects_found = tuple()
    #         camera_direction = None
    #     else:
    #         srobot_old = self.belief.b(self.robot_id).mpe()
    #         robot_pose = srobot_old.pose
    #         objects_found = srobot_old.objects_found
    #         camera_direction = srobot_old.camera_direction
    #     topo_map = self.generate_topo_map(combined_dist,
    #     self.topo_map = topo_map
    #     if not init:
    #         self.policy_model.update(topo_map)
    #     topo_nid = topo_map.closest_node(*robot_pose[:2])
    #     robot_state = RobotStateTopo(self.robot_id,
    #                                  robot_pose,
    #                                  objects_found,
    #                                  camera_direction,
    #                                  topo_nid)
    #     if init:
    #         return robot_state
    #     else:
    #         self.belief.set_object_belief(
    #             self.robot_id, pomdp_py.Histogram({robot_state: 1.0}))



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
