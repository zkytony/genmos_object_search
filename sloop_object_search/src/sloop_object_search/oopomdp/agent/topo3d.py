import random
import pomdp_py

from . import belief
from .basic3d import MosAgentBasic3D
from ..models.octree_belief import OctreeDistribution, OctreeBelief, Octree
from ..models.topo_map import TopoNode, TopoMap, TopoEdge
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils.algo import PriorityQueue
from sloop_object_search.utils import open3d_utils

class MosAgentTopo3D(MosAgentBasic3D):
    """A 3D MosAgent whose action space is not basic axis-based
    primitive movements, but based on a topological graph, where
    each node is a position in the 3D search region that the robot
    can reach. Other aspects"""
    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None):
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                belief_config=self.agent_config["belief"])
        robot_pose = init_robot_pose_dist.mean
        topo_map = _sample_topo_graph3d(init_object_beliefs,
                                        robot_pose,
                                        self.search_region,
                                        self.reachable)
        open3d_utils.draw_topo_graph3d(topo_map, self.search_region,
                                       object_beliefs=init_object_beliefs)
        raise NotImplementedError("good for now")

    def reachable(self, pos):
        """A position is reachable if it is a valid
        voxel and it is not occupied. Assume 'pos' is a
        position at the ground resolution level"""
        # todo: temporary; have some buffer zone to occupancy
        res = 2
        pos = Octree.increase_res(pos, 1, res)
        return self.search_region.octree_dist.octree.valid_voxel(*pos, res)\
            and not self.search_region.occupied_at(pos, res=res)


def _sample_topo_graph3d(init_object_beliefs,
                         init_robot_pose,
                         search_region,
                         reachable_func,
                         num_nodes=10,
                         num_node_samples=1000,
                         degree=(3,5),
                         sep=4.0,
                         rnd=random,
                         score_res=8,
                         score_thres=0.3):
    """
    The algorithm: sample uniformly from search region
    candidate robot positions. Obtain cumulative object
    belief at the sample position but higher resolution
    level, and use the belief as the sample's priority.
    Finally, retain N number of samples according to
    this priority.

    score_res: The resolution level where the object
    belief is used for scoring.

    score_thres: nodes kept must have normalized score
    in the top X% where X is score_thres.
    """
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
    region = search_region.octree_dist.region
    origin, w, l, h = region
    candidate_positions = set([init_robot_pose[:3]])
    candidate_scores = []  # list of (pos, score) tuples
    min_score = float("inf")
    max_score = float("-inf")
    for i in range(num_node_samples):
        # uniformly sample candidate positions
        x = random.uniform(origin[0], origin[0]+w)
        y = random.uniform(origin[1], origin[1]+l)
        z = random.uniform(origin[2], origin[2]+h)
        pos = (x,y,z)
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
            priority_score = 0
            for objid in init_object_beliefs:
                if not isinstance(init_object_beliefs[objid], OctreeBelief):
                    raise ValueError("topo graph3d requires object beliefs to be OctreeBelief")
                b_obj = init_object_beliefs[objid]
                priority_score += b_obj.octree_dist.prob_at(
                    *Octree.increase_res(pos, 1, score_res), score_res)
            candidate_scores.append((pos, priority_score))
            min_score = min(min_score, priority_score)
            max_score = max(max_score, priority_score)

    # Now, we make a priority queue, supply it with normalized scores
    candidates_pq = PriorityQueue()
    candidates_pq.push(init_robot_pose[:3], float('-inf'))  # will always include current robot pose
    for pos, score in candidate_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        if norm_score < score_thres:
            continue
        candidates_pq.push(pos, -norm_score)  # because smaller value has larger priority
    positions = []
    while not candidates_pq.isEmpty() and len(positions) < num_nodes:
        positions.append(candidates_pq.pop())

    # The following is modified based on _sample_topo_map in topo2d
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
        new_neighbors = list(sorted(
            candidates,
            key=lambda pos: math_utils.euclidean_dist(pos, nodes[nid].pos)))[:degree_needed]
        for nbpos in new_neighbors:
            nbnid = pos_to_nid[nbpos]
            if nbnid not in _conns or len(_conns[nbnid]) < degree_range[1]:
                _conns[nid].add(nbnid)
                if nbnid not in _conns:
                    _conns[nbnid] = set()
                _conns[nbnid].add(nid)

                eid = len(edges) + 1000
                edges[eid] = TopoEdge(eid,
                                      nodes[nid],
                                      nodes[nbnid],
                                      {"length": math_utils.euclidean_dist(
                                          nodes[nid].pos, nodes[nbnid].pos)})
    if len(edges) == 0:
        edges[0] = TopoEdge(0, nodes[next(iter(nodes))], None, [])

    topo_map = TopoMap(edges)
    # Verification
    for nid in topo_map.nodes:
        assert len(topo_map.edges_from(nid)) <= degree_range[1]

    return topo_map
