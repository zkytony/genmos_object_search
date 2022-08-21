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
    def __init__(self, x, y, z, res, parent=None, leaf=True, default_val=DEFAULT_VAL):
        """
        DEF: node v represents a voxel centered at the position (x, y, z)
        at resolution r d , where d is the depth for node v. The voxel covers a volume
        of size (r d )^3 . Node v has 8 children that subdivide this volume into
        equal-sized cubes at resolution r d /2. The finest resolution is 1.

        The resolution means how many ground-level cubes (along any dimension) is
        covered by the coordinate (x,y,z). So it should be 1, 2, 4, 8, etc.

        default_val: the default value of a node at the ground level. Use 'value()'
            to get the value of this node.
        """
        # value of the node is stored in the parent. If parent
        # is None, then the value is the default, scaled by resolution.
        self.pos = (x,y,z)
        self.res = res
        self.parent = parent
        self.leaf = leaf
        self._default_val = default_val
        if res > 1:
            self.children = {}# pos: (DEFAULT_VAL, None)
                             # for pos in OctNode.child_poses(x,y,z,res)}  # map from pos to (val, child)
        else:
            self.children = None
            self.val = self._default_val

    def __str__(self):
        num_children = 0 if self.children is None else len(self.children)
        return "OctNode(%s, %d| %.3f)->%d" % (str(self.pos), self.res, self.value(), num_children)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((*self.pos, self.res))

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
                    sum_children_vals += math.exp(self._default_val)*((8-len(self.children))*child_coverage)
                else:
                    sum_children_vals += self._default_val*((8-len(self.children))*child_coverage)
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
                    return self._default_val + math.log((self.res//2)**3)
                else:
                    return self._default_val*((self.res//2)**3)
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
    def __init__(self, dimensions, default_val=DEFAULT_VAL):
        """
        Creates an octree for the given object id, covering volume of
        given dimensions (w,l,h). The depth of the tree is inferred
        from the dimensions, which must satisfy w==l==h and w is power of 2.

        default_val: the default value in a ground octnode
        """
        w,l,h = dimensions
        # requires cubic dimension, power of 2
        assert w == l and l == h and math.log(w, 2).is_integer(),\
            "dimensions must be equal and power of 2; Got (%d, %d, %d)" % (w,l,h)
        dmax = int(round(math.log(w*l*h, 2)))
        self.depth = dmax
        self.root = OctNode(0, 0, 0, w)
        self.dimensions = dimensions
        self._default_val = default_val

    def valid_resolution(self, res):
        return math.log(res, 2).is_integer()\
            and res <= self.dimensions[0]

    def valid_voxel(self, x, y, z, res):
        if res > self.dimensions[0]:
            # Invalid resolution level, too big
            return False
        else:
            return 0 <= x*res < self.dimensions[0]\
                and 0 <= y*res < self.dimensions[0]\
                and 0 <= z*res < self.dimensions[0]

    def __hash__(self):
        return hash((*self.dimensions, self.depth, self.root))

    def __eq__(self, other):
        if not isinstance(other, Octree):
            return False
        else:
            return self.dimensions == other.dimensions\
                and self.normalizer == other.normalizer\
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
        """Note: this method does not allow setting value of this added node.
        To handle that properly, you should use OctreeDistribution.__setitem__"""
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
                child = OctNode(xr, yr, zr, next_res,
                                parent=node, default_val=self._default_val)
                node.add_child(child)
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
        The return format: [(x*r,y*r,z*r,r,v)...] (note the coordinates
        are at the ground level.). Note 'v' is the node value."""
        collector = []
        self._collect_plotting_voxels_helper(self.root, collector)
        return collector

    def _collect_plotting_voxels_helper(self, node, collector):
        if node.leaf:
            collector.append((node.pos[0]*node.res,
                              node.pos[1]*node.res,
                              node.pos[2]*node.res,
                              node.res,
                              node.value()))
        else:
            for pos in OctNode.child_poses(*node.pos, node.res):
                if pos not in node.children\
                   or node.children[pos][1] is None:
                        res = node.res // 2
                        collector.append((pos[0]*res,
                                          pos[1]*res,
                                          pos[2]*res,
                                          res,
                                          node.get_val(pos)))  # child pos and resolution
                else:
                    child = node.children[pos][1]
                    self._collect_plotting_voxels_helper(child, collector)

    @staticmethod
    def increase_res(point, r1, r2):
        """Given a point (x,y,z) at resolution r1,
        returns a new point (x', y', z') expressed
        in (0-based) coordinates at resolution r2,
        where r2 >= r1.

        For example, if r1=1, r2=2, then if point
        is (10, 1, 5), then the output is (5, 0, 2).
        Only works exactly if r1 <= r2; If r1 > r2,
        will return """
        x,y,z = point
        if r1 > r2:
            raise ValueError("requires r1 <= r2")
        return (x // (r2 // r1), y // (r2 // r1), z // (r2 // r1))
