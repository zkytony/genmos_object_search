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

import math

LOG = False  # the values are log-space probabilities.
if LOG:
    DEFAULT_VAL=0.
else:
    DEFAULT_VAL=1.

class OctNode:
    def __init__(self, x, y, z, res, parent=None, leaf=True):
        """
        DEF: node v represents a voxel centered at the position (x, y, z)
        at resolution r d , where d is the depth for node v. The voxel covers a volume
        of size (r d )^3 . Node v has 8 children that subdivide this volume into
        equal-sized cubes at resolution r d /2. The finest resolution is 1.

        The resolution means how many ground-level cubes (along any dimension) is
        covered by the coordinate (x,y,z).
        """
        # value of the node is stored in the parent. If parent
        # is None, then the value is the default, scaled by resolution.
        self.pos = (x,y,z)
        self.res = res
        self.parent = parent
        self.leaf = leaf
        if res > 1:
            self.children = {}# pos: (DEFAULT_VAL, None)
                             # for pos in OctNode.child_poses(x,y,z,res)}  # map from pos to (val, child)
        else:
            self.children = None
            self.val = DEFAULT_VAL
            
    def __str__(self):
        num_children = 0 if self.children is None else len(self.children)
        return "OctNode(%s, %d| %.3f)->%d" % (str(self.pos), self.res, self.value(), num_children)
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((*self.pos, self.res))

    @property
    def pose(self):
        return self.pos
    
    def add_child(self, child):
        if self.children is None:
            raise ValueError("Ground node cannot have children")
        self.children[child.pos] = (child.value(), child)
        child.parent = self
        # no longer a leaf node
        self.leaf = False

    def child_at(self, pos):
        if self.children is None:
            return None
        else:
            if pos in self.children:
                return self.children[pos][1]
            else:
                return None

    def has_child(self, pos):
        return self.child_at(pos) != None

    def value(self):
        """The value of this node is the sum of its children"""
        if self.children is not None:
            sum_children_vals = 0  # this is not in log space
            if len(self.children) > 0:
                if LOG:
                    # children value is in log space.
                    sum_children_vals += sum([math.exp(self.children[p][0]) for p in self.children])
                else:
                    sum_children_vals += sum([self.children[p][0] for p in self.children])
            if len(self.children) < 8:
                child_coverage = (self.res // 2)**3
                if LOG:
                    # DEFAULT_VAL is in log space.
                    sum_children_vals += math.exp(DEFAULT_VAL)*((8-len(self.children))*child_coverage)
                else:
                    sum_children_vals += DEFAULT_VAL*((8-len(self.children))*child_coverage)
            if LOG:
                return math.log(sum_children_vals)
            else:
                return sum_children_vals
        else:
            # This node is ground level. It stores its own value.
            assert self.res == 1
            return self.val

    def set_val(self, child_pos, val, child=None):
        if child_pos is not None:
            self.children[child_pos] = (val, child)
        else:
            assert self.res == 1            
            self.val = val

    def get_val(self, child_pos):
        if child_pos is not None:
            if child_pos not in self.children:
                # return default value
                if LOG:
                    return DEFAULT_VAL + math.log((self.res//2)**3)
                else:
                    return DEFAULT_VAL*((self.res//2)**3)
            else:
                return self.children[child_pos][0]
        else:
            assert self.res == 1  # a ground octnode
            return self.val

    @staticmethod
    def child_poses(x, y, z, res):
        """return a set of poses of the children of node
        centered at (x, y, z, res); Doesn't matter if the child node exists"""
        if res > 1:
            xc, yc, zc = x*2, y*2, z*2
            poses = set({
                (xc + dx, yc + dy, zc + dz)
                for dx in range(2)
                for dy in range(2)
                for dz in range(2)
            })
            return poses
        else:
            return None
        
class Octree:
    def __init__(self, objid, dimensions):
        """
        Creates an octree for the given object id, covering volume of
        given dimensions (w,l,h). The depth of the tree is inferred
        from the dimensions, which must satisfy w==l==h and w is power of 2.
        """
        self.objid = objid
        w,l,h = dimensions
        # requires cubic dimension, power of 2
        assert w == l and l == h and math.log(w, 2).is_integer(),\
            "dimensions must be equal and power of 2; Got (%d, %d, %d)" % (w,l,h)
        dmax = int(round(math.log(w*l*h, 2)))
        self.depth = dmax
        self.root = OctNode(0, 0, 0, w)
        self.dimensions = dimensions

        # normalizer; we only need one normalizer at the ground level.
        # NOTE that the normalizer is not in log space.
        if LOG:
            # the default value is in log space; So we have to convert it.
            self._normalizer = (w*l*h)*math.exp(DEFAULT_VAL)
        else:
            self._normalizer = (w*l*h)*DEFAULT_VAL

        # stores locations where occupancy was once recorded
        self._known_voxels = {}
        next_res = self.root.res
        while next_res >= 1:        
            # for efficiency; weights are not in log space.
            self._known_voxels[next_res] = {}  # voxel pose -> weight (not log space)
            next_res = next_res // 2            

    def valid_resolution(self, res):
        return res in self._known_voxels

    def known_voxels(self, res):
        """return set of voxel poses at resolution level"""
        if res not in self._known_voxels:
            raise ValueError("resolution invalid %d" % res)
        return self._known_voxels[res] #['voxels']
    
    def update_node_weight(self, x, y, z, res, value):
        if LOG:
            # value is in log space
            self._known_voxels[res][(x,y,z)] = math.exp(value)
        else:
            self._known_voxels[res][(x,y,z)] = value

    def node_weight(self, x, y, z, res):
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

    def __hash__(self):
        return hash((*self.dimensions, self.depth, self.root, self.objid))

    def __eq__(self):
        if not isinstance(other, Octree):
            return False
        else:
            return self.objid == other.objid\
                and self.dimensions == other.dimensions\
                and self.normalizer == other.normalizer\
                and self._known_voxels == other._known_voxels\
                and self._once_occupied == other._once_occupied\
                and self.depth == other.depth\
                and self.root == other.root

    def print_tree_helper(self, val, node, depth, max_depth=None):
        if max_depth is not None and depth >= max_depth and depth >= self.depth:
            return
        print("%s%s" % ("    "*depth, node))
        if node.children is not None:
            for pos in node.children:
                val, c = node.children[pos]
                self.print_tree_helper(val, c, depth+1, max_depth=max_depth)

    def print_tree(self, max_depth=None):
        self.print_tree_helper(self.root.value, self.root, 0, max_depth=max_depth)

    def add_node(self, x, y, z, res):
        if x*res >= self.dimensions[0]\
           or y*res >= self.dimensions[1]\
           or z*res >= self.dimensions[2]:
            raise ValueError("Invalid voxel position %s" % str((x,y,z,res)))
        
        node = self.root
        next_res = self.root.res // 2  # resolution
        while next_res >= res:
            # everything should already be integers
            xr = x // (next_res // res)
            yr = y // (next_res // res)
            zr = z // (next_res // res)
            if not node.has_child((xr, yr, zr)):
                child = OctNode(xr, yr, zr, next_res, parent=node)
                node.add_child(child)
                if LOG:
                    # child.value will be log space probability; But we want to store
                    # natural probability as weights.
                    self._known_voxels[next_res][(xr,yr,zr)] = math.exp(child.value())
                else:
                    # child.value will be natural space probability.
                    self._known_voxels[next_res][(xr,yr,zr)] = child.value()
            node = node.child_at((xr,yr,zr))
            next_res = node.res // 2
        return node

    def get_node(self, x, y, z, res):
        node = self.root
        next_res = self.root.res // 2  # resolution
        while next_res >= res:
            # everything should already be integers
            xr = x // (next_res // res)
            yr = y // (next_res // res)
            zr = z // (next_res // res)
            node = node.child_at((xr,yr,zr))
            if node is None:
                return None
            next_res = node.res // 2
        return node

    def get_leaves(self):
        # Returns a list of OctNodes that are leaves.
        all_leaves = []
        self._get_leaves_helper(self.root, all_leaves)
        return all_leaves

    def _get_leaves_helper(self, node, collector):
        if node.leaf:
            collector.append(node)
        else:
            for pos in node.children:
                if node.has_child(pos):
                    self._get_leaves_helper(node.child_at(pos), collector)

    # Visualization
    def collect_plotting_voxels(self):
        """Returns voxel positions and resolutions that should
        be plotted, given an octree. Note that not all
        of these actually exist in the tree as a node.
        The return format: [(x*r,y*r,z*r,r)...] (note the coordinates
        are at the ground level.)"""
        collector = []
        self._collect_plotting_voxels_helper(self.root, collector)
        return collector
    
    def _collect_plotting_voxels_helper(self, node, collector):
        if node.leaf:
            collector.append((node.pos[0]*node.res,
                              node.pos[1]*node.res,
                              node.pos[2]*node.res,
                              node.res))
        else:
            for pos in OctNode.child_poses(*node.pos, node.res):
                if pos not in node.children\
                   or node.children[pos][1] is None:
                        res = node.res // 2
                        collector.append((pos[0]*res,
                                          pos[1]*res,
                                          pos[2]*res, res))  # child pos and resolution
                else:
                    child = node.children[pos][1]
                    self._collect_plotting_voxels_helper(child, collector)            
                
