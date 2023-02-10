import numpy as np
import uuid
from collections import deque
from genmos_object_search.utils.colors import lighter
from genmos_object_search.utils.math import euclidean_dist
from genmos_object_search.utils.graph import Node, Graph, Edge

class TopoNode(Node):
    """TopoNode is a node on the topological graph."""

    def __init__(self, node_id, pos):
        """
        pos (tuple): The position on the underlying
                     (metric or grid) space where this topo node
                     can be grounded.
        """
        super().__init__(node_id, pos)
        self._coords = pos

    @property
    def pos(self):
        return self.data

    def __str__(self):
        return "n{}@{}".format(self.id, self.pos)

    def __repr__(self):
        return "TopoNode({})".format(str(self))


class TopoEdge(Edge):
    def __init__(self, id, node1, node2, nav_info):
        super().__init__(id, node1, node2, data=nav_info)

    @property
    def nav_info(self):
        return self.data

    @property
    def nav_path(self):
        return self.nav_info.get('path', None)

    @property
    def nav_length(self):
        """length of naviagtion path"""
        return self.nav_info.get('length', float('inf'))

    @property
    def attrs(self):
        return {"nav_path": self.nav_path,
                "nav_length": len(self.nav_length)}


class TopoMap(Graph):

    """To create a TopoMap,
    construct a mapping called e.g. `edges` that maps from edge id to Edge,
    and do TopoMap(edges)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_closest = {}
        self._cache_shortest_path = {}
        # A topo map will be uniquely identified by this hashcode
        self._hashcode = uuid.uuid4().hex

    @property
    def hashcode(self):
        return self._hashcode

    def __hash__(self):
        return hash(self._hashcode)

    def __eq__(self, other):
        if isinstance(other, TopoMap):
            return self.hashcode == other.hashcode
        else:
            return False

    def closest_node(self, point):
        """Given a point find the node that is closest to this point.
        """
        if point in self._cache_closest:
            return self._cache_closest[point]
        else:
            nid = min(self.nodes,
                       key=lambda nid: euclidean_dist(self.nodes[nid].pos, point))
            self._cache_closest[point] = nid
            return nid

    def edge_between(self, nid1, nid2):
        edges = self.edges_between(nid1, nid2)
        if edges is None:
            return None
        if len(edges) > 1:
            raise ValueError("There are multiple edges between nodes {} and {}. Unexpected."\
                             .format(nid1, nid2))
        return next(iter(edges))

    def navigable(self, nid1, nid2):
        # DFS find path from nid1 to nid2
        stack = deque()
        stack.append(nid1)
        visited = set()
        while len(stack) > 0:
            nid = stack.pop()
            if nid == nid2:
                return True
            for neighbor_nid in self.neighbors(nid):
                if neighbor_nid not in visited:
                    stack.append(neighbor_nid)
                    visited.add(neighbor_nid)
        return False

    def shortest_path(self, src, dst):
        if (src, dst) in self._cache_shortest_path:
            return self._cache_shortest_path[(src, dst)]
        else:
            path = super().shortest_path(src, dst, lambda e: e.grid_dist)
            self._cache_shortest_path[(src, dst)] = path
            return path
