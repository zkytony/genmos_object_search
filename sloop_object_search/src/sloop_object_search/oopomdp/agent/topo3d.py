import random
import pomdp_py

from . import belief
from .basic3d import MosAgentBasic3D
from ..domain.state import RobotStateTopo
from ..domain.observation import RobotObservation, RobotObservationTopo
from ..models.octree_belief import OctreeDistribution, OctreeBelief, Octree
from ..models.topo_map import TopoNode, TopoMap, TopoEdge
from ..models.policy_model import PolicyModelTopo
from ..models.transition_model import RobotTransTopo3D
from ..models.observation_model import RobotObservationModelTopo
from .common import MosAgent, SloopMosAgent, init_object_transition_models
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
        self.topo_map = self.generate_topo_map(init_object_beliefs, robot_pose)

        init_topo_nid = self.topo_map.closest_node(robot_pose[:3])
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
        robot_trans_model = RobotTransTopo3D(self.robot_id, target_ids,
                                             self.topo_map, self.detection_models,
                                             no_look=self.no_look,
                                             default_camera_direction=self.default_forward_direction)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        policy_model = PolicyModelTopo(target_ids,
                                       robot_trans_model,
                                       no_look=self.no_look)
        return transition_model, policy_model

    def generate_topo_map(self, object_beliefs, robot_pose, debug=False):
        """object_beliefs: objid->OctreeBelief.
        robot_pose: a 7-tuple"""
        topo_map = _sample_topo_graph3d(object_beliefs,
                                        robot_pose,
                                        self.search_region,
                                        self.reachable,
                                        self.topo_config)
        if debug:
            open3d_utils.draw_topo_graph3d(topo_map, self.search_region,
                                           object_beliefs=object_beliefs)
        return topo_map

    @property
    def topo_config(self):
        return self.agent_config.get("topo", {})

    def reachable(self, pos):
        """A position is reachable if it is a valid
        voxel and it is not occupied. Assume 'pos' is a
        position at the ground resolution level"""
        # res_buf: blows up the voxel at pos to keep some distance to obstacles
        # It will affect where the topological graph nodes are placed with respect
        # to obstacles.
        res = self.topo_config.get("res_buf", 4)
        pos = Octree.increase_res(pos, 1, res)
        return self.search_region.octree_dist.octree.valid_voxel(*pos, res)\
            and not self.search_region.occupied_at(pos, res=res)

    def update_robot_belief(self, observation, action=None, **kwargs):
        current_srobot_mpe = self.belief.mpe().s(self.robot_id)
        super().update_robot_belief(observation, action=action,
                                    robot_state_class=RobotStateTopo,
                                    topo_nid=current_srobot_mpe.topo_nid,
                                    topo_map_hashcode=current_srobot_mpe.topo_map_hashcode)

    def update_belief(self, observation, action=None, debug=False, **kwargs):
        """returns auxiliary info generated from belief update (for
        debugging or visualization purposes"""
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
                    object_beliefs, robot_pose, debug=debug)
        return _aux

    def update_topo_map(self, topo_map, robot_observation):
        """
        we expect robot_observation to be RobotObservation which
        contains a RobotLocalization; This is checked by update_robot_belief
        """
        self.topo_map = topo_map
        self.policy_model.update(topo_map)
        # Now, we need to update the robot belief because new topo map leads
        # to new topo nid and hash
        topo_nid = self.topo_map.closest_node(*robot_observation.loc)
        super().update_robot_belief(observation, action=action,
                                    robot_state_class=RobotStateTopo,
                                    topo_nid=topo_nid,
                                    topo_map_hashcode=self.topo_map.hashcode)

    def should_resample_topo_map(self, object_beliefs):
        # Get an estimate of total belief captured by the current topological nodes
        zone_res = self.topo_config.get("sampling", {}).get("zone_res", 8)
        resample_prob_thres = self.topo_config.get("sampling", {}).get("zone_res", 0.4)
        total_prob = 0
        zones_covered = set()  # set of zones whose area's probability has been considered
        for nid in self.topo_map.nodes:
            pos = self.topo_map.nodes[nid].pos
            zone_pos = Octree.increase_res(pos, 1, zone_res)
            if zone_pos in zones_covered:
                continue
            prob = _compute_combined_prob_around(pos, object_beliefs, zone_res)
            total_prob += prob
            zones_covered.add(zone_pos)
        # total_prob should be a normalized probability
        assert 0 <= total_prob <= 1
        return total_prob < resample_prob_thres


def _compute_combined_prob_around(pos, object_beliefs, zone_res=8):
    """Given a position 'pos', and object beliefs, and a resolution level for the
    zone which the belief around the position is considered, return a normalized
    probability that combines the beliefs over all objects within that zone.
    """
    comb_prob = 0
    for objid in object_beliefs:
        if not isinstance(object_beliefs[objid], OctreeBelief):
            raise ValueError("topo graph3d requires object beliefs to be OctreeBelief")
        b_obj = object_beliefs[objid]
        comb_prob += b_obj.octree_dist.prob_at(
            *Octree.increase_res(pos, 1, zone_res), zone_res)
    # because 'prob_at' returns normalized prob for each object, the normalizer
    # of the combinations is just the number of objects.
    return comb_prob / len(object_beliefs)


def _sample_topo_graph3d(init_object_beliefs,
                         init_robot_pose,
                         search_region,
                         reachable_func,
                         topo_config={}):
    """
    The algorithm: sample uniformly from search region
    candidate robot positions. Obtain cumulative object
    belief at the sample position but higher resolution
    level, and use the belief as the sample's priority.
    Finally, retain N number of samples according to
    this priority.

    zone_res: The resolution level where the object
    belief is used for scoring.

    score_thres: nodes kept must have normalized score
    in the top X% where X is score_thres.
    """
    # parameters
    num_nodes = topo_config.get("num_nodes", 10)
    # number of samples when sampling topo graph nodes.
    num_node_samples = topo_config.get("num_node_samples", 1000)
    # Minimum and maximum out degrees at each node
    degree = topo_config.get("degree", (3,5))
    # Minimum separation between nodes (in POMDP scale)
    sep = topo_config.get("sep", 4.0)
    rnd = topo_config.get("random", random)
    # Determins the size of the area around a position to be considered
    # for estimating score at that position.
    zone_res = topo_config.get("zone_res", 8)
    # Threshold of the probability estimated at a position for accepting that
    # position as a topo graph node position candidate.
    prob_pos_thres = topo_config.get("prob_pos_thres", 0.3)

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
            prob_pos =\
                _compute_combined_prob_around(pos, init_object_beliefs, zone_res=zone_res)
            candidate_scores.append((pos, prob_pos))

    positions = [init_robot_pose[:3]]
    positions.extend(list(c[0] for c in
                          reversed(sorted(candidate_scores, key=lambda c: c[1]))
                          if c[1] > prob_pos_thres))
    if num_nodes > len(positions):
        positions = positions[:num_nodes]

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
