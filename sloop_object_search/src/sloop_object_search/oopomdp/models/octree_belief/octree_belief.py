# Copyright 2022 Kaiyu Zheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pomdp_py
import math
import random
import numpy as np
import sys
import copy
import sloop_object_search.utils.math as util
from sloop_object_search.oopomdp.domain.state import ObjectState
from .octree import LOG, DEFAULT_VAL, OctNode, Octree
from sloop_object_search.oopomdp.domain.observation import FovVoxels, Voxel

class OctreeDistribution(pomdp_py.GenerativeDistribution):
    """A distribution over 3D grid cells. A random variable X
    following this distribution has a value space G^l (the 3D grids)
    at a given resolution level, l,

    Pr(X=x^l) = Val(g_x^l) / Normalizer

    where g_x^l is the octree node at the grid with location x at
    resolution level l.

    The value stored at a node g_x^l represents an unnormalized
    probability that X=x^l, and it is the sum of the values of
    the children of node g_x^l.

    An octree node can be referenced by the voxel it captures, as a tuple: (x, y, z, res)

    Different from OctreeBelief, which is used specifically for
    representing distribution over ObjectState, this distribution
    is general.
    """
    def __init__(self, octree, default_val=DEFAULT_VAL):
        """For alpha, beta, gamma, refer to ObjectObservationModel."""
        self._octree = octree
        self._gamma = default_val

        # world dimensions
        w, l, h = octree.dimensions

        # normalizer; we only need one normalizer at the ground level.
        # NOTE that the normalizer is not in log space.
        if LOG:
            # the default value is in log space; So we have to convert it.
            self._normalizer = (w*l*h)*math.exp(self._gamma)
        else:
            self._normalizer = (w*l*h)*self._gamma

        # stores locations where occupancy was once recorded (cache)
        self._known_voxels = {}
        next_res = self.octree.root.res
        while next_res >= 1:
            # for efficiency; weights are not in log space.
            self._known_voxels[next_res] = {}  # voxel pose -> weight (not log space)
            next_res = next_res // 2

    @property
    def octree(self):
        return self._octree

    def known_voxels(self, res):
        """return set of voxel poses at given resolution"""
        if res not in self._known_voxels:
            raise ValueError("resolution invalid %d" % res)
        return self._known_voxels[res] #['voxels']

    def update_node_weight_cache(self, x, y, z, res, value):
        if LOG:
            # value is in log space
            self._known_voxels[res][(x,y,z)] = math.exp(value)
        else:
            self._known_voxels[res][(x,y,z)] = value

    def node_weight_in_cache(self, x, y, z, res):
        # Note that the weights in known_voxels are not in log space.
        if res in self._known_voxels and (x,y,z) in self._known_voxels[res]:
            return self._known_voxels[res][(x,y,z)]
        else:
            return None

    def update_normalizer(self, old_value, value):
        if LOG:
            self._normalizer += (math.exp(value) - math.exp(old_value))
        else:
            self._normalizer += (value - old_value)

    def normalized_probability(self, node_value):
        """Given the value of some node, which represents unnormalized proability,
        returns the normalized probability, properly converted to log space
        depending on the setting of LOG."""
        if LOG:
            # node_value is in log space.
            return node_value - math.log(self._normalizer)  # the normalizer property takes care of log space issue.
        else:
            # node value is not in log space.
            return node_value / self._normalizer

    def __getitem__(self, voxel):
        """voxel: a tuple (x, y, z, res)"""
        if type(voxel) != tuple and len(voxel) != 4:
            raise ValueError("Requires voxel to be a tuple (x,y,z,res)")
        x,y,z = voxel[:3]
        res = voxel[3]
        return self._probability(x, y, z, res)

    def prob_at(self, x, y, z, res, fast=True):
        return self._probability(x, y, z, res, fast=fast)

    def _probability(self, x, y, z, res, fast=True):
        """
        Probability of object present at voxel centered at the position (x, y, z)
        with resolution `res`. See OctNode definition.

        Example (5,5,5,1) is different from (5,5,5,2).
        """
        if not self._octree.valid_resolution(res):
            raise ValueError("Resolution %d is not defined in the octree" % res)

        if fast:
            # Note that the weights in known_voxels are not in log space.
            weight = self.node_weight_in_cache(x, y, z, res)
            if weight is not None:
                if LOG:
                    weight = math.log(weight)  # convert to log space
                return self.normalized_probability(weight)
            else:
                prob_one_voxel = self.normalized_probability(self._gamma)
                if LOG:
                    # prob_one_voxel is in log space.
                    return math.log(math.exp(prob_one_voxel)*((res)**3))
                else:
                    return prob_one_voxel * ((res)**3)
        else:
            node = self._octree.root
            next_res = self._octree.root.res // 2  # resolution
            while next_res >= res:
                xr = x // (next_res // res)
                yr = y // (next_res // res)
                zr = z // (next_res // res)

                if node.has_child((xr, yr, zr)):
                    next_res = next_res // 2
                    node = node.child_at((xr, yr, zr))
                else:
                    # Has not encountered this position. Thus the voxel
                    # has not been observed. P(v|s',a)=gamma; need to account
                    # for the resolution.
                    prob_one_voxel = self.normalized_probability(self._gamma)
                    if LOG:
                        # prob_one_voxel is in log space.
                        return math.log(math.exp(prob_one_voxel)*((res)**3))
                    else:
                        return prob_one_voxel * ((res)**3)
            # Have previously observed this position and there's a node for it.
            # Use the node's value to compute the probability
            return self.normalized_probability(node.value())

    def __setitem__(self, voxel, value):
        """
        This can only happen for voxels at ground level.
        The value is unnormalized probability. Will treat as in
        log space if LOG is true.
        """
        if type(voxel) != tuple and len(voxel) != 4:
            raise ValueError("Requires voxel to be a tuple (x,y,z,res)")
        x,y,z = voxel[:3]
        res = voxel[3]
        node = self._octree.add_node(x,y,z,1)
        old_value = node.get_val(None)
        node.set_val(None, value)
        self.update_normalizer(old_value, value)
        self.backtrack(node)

    def assign(self, voxel, value):
        """
        This can happen for voxels at different resolution levels.
        The value is unnormalized probability. Will treat as in
        log space if LOG is true.

        Note: This will make child voxels a uniform distribution;
        Should only be used for setting prior.
        """
        if type(voxel) != tuple and len(voxel) != 4:
            raise ValueError("Requires voxel to be a tuple (x,y,z,res)")
        res = voxel[3]
        if res >= self._octree.root.res:
            raise ValueError("Resolution too large for assignment (%d>=%d)"
                             % (res, self._octree.root.res))

        x,y,z = voxel[:3]
        node = self._octree.add_node(x, y, z, res)
        # set the value of the node
        if res > 1:
            for pos in OctNode.child_poses(x, y, z, res):
                if not node.has_child(pos):
                    child = None
                else:
                    child = node.child_at(pos)
                if LOG:
                    node.set_val(pos, value - math.log(8), child=child)
                else:
                    node.set_val(pos, value / 8, child=child)
        else:
            # node is at ground level so it has no children. Just set its value
            node.set_val(None, value)
        old_value = node.parent.get_val(node.pos)
        # Make sure the parent agrees with the value of the node
        node.parent.set_val((x,y,z), value, child=node)
        self.update_normalizer(old_value, value)
        self.backtrack(node)
        self._propagate(node)

    def random(self, res=1):
        """Returns a voxel position (x,y,z) at resolution 'res'."""
        voxel_pos = self._random_path(res, argmax=False)
        return voxel_pos

    def mpe(self, res=1):
        voxel_pos = self._random_path(res, argmax=True)
        return voxel_pos

    def random_child(self, pos=None, res=None, argmax=False, node=None):
        """Returns a position (x,y,z) that is a location considered the 'child' of
        the node located at `node`(x,y,z,res), according to the distribution of the
        children of this node. If such child has not been added to the octree,
        one located at a uniformly chosen child position will be returned."""
        if node is None and (pos is None or res is None):
            raise ValueError("Either provide node, or provide pos&res.")
        if node is None:
            node = self._octree.get_node(*pos, res)
            if node is None:
                # node at (pos, res) has not been added to the tree. So just
                # return a child pos uniformly at random.
                child_poses = list(OctNode.child_poses(*pos, res))
                chosen_pos = random.choice(child_poses)
                return chosen_pos

        # Choose a child position
        child_poses = list(OctNode.child_poses(*node.pos, node.res))
        child_vals = np.array([node.get_val(child_pos)
                               for child_pos in child_poses])
        if argmax:
            chosen_pos_index = np.argmax(child_vals)
            chosen_pos = child_poses[chosen_pos_index]
        else:
            if sys.version_info[0] >= 3 and sys.version_info[1] >= 6:
                chosen_pos = random.choices(child_poses, weights=child_vals, k=1)[0]
            else:
                chosen_pos_index = np.random.choice(np.arange(len(child_poses)), 1, p=child_vals)[0]
                chosen_pos = child_poses[chosen_pos_index]
        return chosen_pos

    def random_ground_child_pos(self, pos, res):
        """Returns (x,y,z) of a ground voxel which is covered by the
        given supervoxel (*pos, res) with probability encoded in the octree."""
        # Takes a path from node at (*pos, res) to target_res=1.
        if res <= 1:
            raise ValueError("The given voxel is already a ground voxel.")
        node = self._octree.get_node(*pos, res)
        if node is None:
            return self._random_unif_child(pos, res, 1)
        else:
            return self._random_path_helper(node, 1)

    def _random_path(self, target_res, argmax=False):
        """Takes a random path from root downwards, till hitting a node
        with resolution equal to `target_res`. Returns the pose (x,y,z)
        of the end point node (which would be at the target resolution."""
        if target_res < 1:
            raise ValueError("Path cannot go lower than resolution 1.")
        return self._random_path_helper(self._octree.root, target_res, argmax=argmax)

    def _random_path_helper(self, node, target_res, argmax=False):
        if node.res < target_res:
            raise ValueError("Invalid target res %d. It is not a power of 2." % target_res)
        elif node.res == target_res:
            return node.pos
        else:
            chosen_pos = self.random_child(node=node, argmax=argmax)
            child = node.child_at(chosen_pos)
            if child is None:
                return self._random_unif_child(chosen_pos, node.res // 2, target_res)
            else:
                return self._random_path_helper(child, target_res)

    def _random_unif_child(self, child_pos, child_res, target_res):
        """Assuming node at (*child_pos, child_res) does not exist in the
        octree. Thus, return a pos at resolution `target_res` that is covered
        by child pos."""
        # The node is already None (i.e. tree doesn't contain this child)
        # Then sample a pos uniformly within.
        gapres = child_res // target_res
        xc, yc, zc = child_pos
        unif_pos = util.uniform(3,
                                [(xc*gapres, (xc+1)*gapres),
                                 (yc*gapres, (yc+1)*gapres),
                                 (zc*gapres, (zc+1)*gapres)])
        return unif_pos


    def backtrack(self, node):
        cur_supernode = node.parent
        cur_node = node
        while cur_supernode is not None:
            # Update the value of child through accessing parent;
            # Update the known_voxels dict as well through update_node_weight_cache
            cur_node_val = cur_node.value()
            cur_supernode.set_val(cur_node.pos, cur_node_val, child=cur_node)
            self.update_node_weight_cache(*cur_node.pos, cur_node.res, cur_node_val)
            cur_node = cur_supernode
            cur_supernode = cur_supernode.parent
        # cur_node is the root node. Also need to call update_node_weight_cache
        assert cur_node.res == self._octree.root.res
        cur_node_val = cur_node.value()
        self.update_node_weight_cache(*cur_node.pos, cur_node.res, cur_node_val)

    def _propagate(self, node):
        """Update the value in octree's known_voxels set, for all children voxels,
        at all resolution levels."""
        self._propagate_helper(*node.pos, node.res, node.value())

    def _propagate_helper(self, x, y, z, res, val):
        """The value is unnormalized probability. Will treat as in
        log space if LOG is true."""
        self.update_node_weight_cache(x, y, z, res, val)
        if res > 1:
            for child_pos in OctNode.child_poses(x, y, z, res):
                if LOG:
                    # want to compute: log(exp(val)/8) = log(exp(val)) - log(8) = x - log(8)
                    self._propagate_helper(*child_pos, res // 2, val - math.log(8))
                else:
                    # want to compute: val / 8
                    self._propagate_helper(*child_pos, res // 2, val / 8)


class OctreeBelief(pomdp_py.GenerativeDistribution):
    """
    OctreeBelief is a belief designed specifically for the
    3D object search problem.

    Each object is associated with an octree belief, separate from others.

    An octree belief has an underlying octree distribution. This handles
    the probability behavior over the 3D space captured by the corresponding
    octree.
    """
    def __init__(self, objid, objclass, octree_dist, default_val=DEFAULT_VAL):
        if not isinstance(octree_dist, OctreeDistribution):
            raise TypeError("octree_dist must be an instance of OctreeDistribution")
        self._objid = objid
        self._objclass = objclass
        self._octree_dist = octree_dist

    @property
    def objid(self):
        return self._objid

    @property
    def octree(self):
        return self._octree_dist.octree

    @property
    def octree_dist(self):
        return self._octree_dist

    def __getitem__(self, object_state):
        if object_state.id != self._objid:
            raise TypeError("Given object state has object id %d\n"\
                            "but this belief is for object id %d"
                            % (object_state.id, self._objid))
        x,y,z = object_state.loc
        res = object_state.res
        return self._octree_dist.prob_at(x, y, z, res)

    def __setitem__(self, object_state, value):
        """
        This can only happen for object state at ground level.
        The value is unnormalized probability. Will treat as in
        log space if LOG is true.
        """
        if object_state.res != 1:
            raise TypeError("Only allow setting the value at ground level.")
        x,y,z = object_state.loc
        self._octree_dist[(x,y,z,object_state.res)] = value

    def assign(self, object_state, value):
        """
        This can happen for object state at different resolution levels.
        The value is unnormalized probability. Will treat as in
        log space if LOG is true.

        Note: This will make child voxels a uniform distribution;
        Should only be used for setting prior.
        """
        if object_state.res >= self.octree.root.res:
            raise ValueError("Resolution too large for assignment (%d>=%d)"
                             % (object_state.res, self._octree.root.res))

        x,y,z = object_state.pose
        self._octree_dist.assign((x,y,z,object_state.res), value)

    def random(self, res=1):
        voxel_pos = self._octree_dist.random(res=res)
        return ObjectState(self._objid, self._objclass, voxel_pos, res=res)

    def mpe(self, res=1):
        voxel_pos = self._octree_dist.mpe(res=res)
        return ObjectState(self._objid, self._objclass, voxel_pos, res=res)


def update_octree_belief(octree_belief, real_action, real_observation,
                         alpha=1000., beta=0., gamma=DEFAULT_VAL):
    """
    For alpha, beta, gamma, refer to ObjectObservationModel.
    real_observation (Observation)
    """
    if not isinstance(real_observation, FovVoxels):
       raise TypeError("Belief update should happen using "\
                       "unfactored observation (type Observation)")

    # Make a copy
    octree_belief = copy.deepcopy(octree_belief)  # ? is this necessary?
    for voxel_pose in real_observation.voxels:
        ## TODO: Should make sure voxel_pose is actually within search space boundary,
        ## which is specified in the octree_belief already.
        voxel = real_observation.voxels[voxel_pose]
        x, y, z = voxel_pose
        node = octree_belief.octree.add_node(x,y,z,1)  # real_observation is ground level.
        val_t = node.get_val(None)  # get value at node.
        if LOG:
            if voxel.label == Voxel.UNKNOWN:
                node.set_val(None, val_t + gamma)
            elif voxel.label == octree_belief.objid:
                node.set_val(None, val_t + alpha)
            else:
                node.set_val(None, val_t + beta)
        else:
            if voxel.label == Voxel.UNKNOWN:
                node.set_val(None, val_t * gamma)
            elif voxel.label == octree_belief.objid:
                # override potential previous belief of free space due to beta=0
                node.set_val(None, val_t * alpha)
            else:
                node.set_val(None, val_t * beta)
        val_tp1 = node.get_val(None)
        octree_belief.octree_dist.update_normalizer(val_t, val_tp1)
        octree_belief.octree_dist.backtrack(node)
    return octree_belief


def init_octree_belief(gridworld, init_robot_state, prior=None):
    """
    Returns dictionary `object_beliefs` mapping from objid to object belief.
    If the object is robot, the belief is a histogram; Otherwise,
    the belief is an octree.

    prior: objid -> {(x,y,z,r) -> value}
    """
    w, l, h = gridworld.width, gridworld.length, gridworld.height
    object_beliefs = {}
    # robot belief is over target objects; therefore, obstacles will
    # not be included in the state.
    for objid in gridworld.target_objects:
        objclass = gridworld.objects[objid].objtype
        octree = Octree((w, l, h))
        octree_belief = OctreeBelief(w, l, h, objid, objclass, octree)
        if prior is not None and objid in prior:
            for x,y,z,r in prior[objid]:
                state = ObjectState(objid, objclass, (x,y,z), res=r)
                octree_belief.assign(state, prior[objid][(x,y,z,r)])
        object_beliefs[objid] = octree_belief
    object_beliefs[gridworld.robot_id] = pomdp_py.Histogram({init_robot_state: 1.0})
    return object_beliefs




class RegionalOctree(Octree):
    """
    This is an octree with a default value of 0 for (ground-level) nodes
    outside of a region, defined either by a box (center, w, h, l), or
    a set of voxels (could be at different resolution levels).
    """
    # def __init__(self,
    pass
