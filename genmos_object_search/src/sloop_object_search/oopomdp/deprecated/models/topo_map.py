import numpy as np
from collections import deque
from genmos_object_search.utils.colors import lighter
from genmos_object_search.utils.math import euclidean_dist
from genmos_object_search.utils.graph import Node, Graph, Edge

class TopoNode(Node):
    """TopoNode is a node on the grid map."""

    def __init__(self, node_id, grid_pos):
        """
        grid_pos (x,y): The grid position this node is located
        search_region_locs (list or set): locations where the
            target object can be for which this node is the closest.
        """
        super().__init__(node_id, grid_pos)
        self._coords = grid_pos

    @property
    def pos(self):
        return self.data

    def __str__(self):
        return "n{}@{}".format(self.id, self.pos)

    def __repr__(self):
        return "TopoNode({})".format(str(self))


class TopoEdge(Edge):
    def __init__(self, id, node1, node2, grid_path):
        super().__init__(id, node1, node2, data=grid_path)

    @property
    def grid_path(self):
        return self.data

    @property
    def grid_dist(self):
        if self.grid_path is None:
            return float('inf')
        else:
            return len(self.grid_path)

    @property
    def nav_length(self):
        return self.grid_dist

    @property
    def attrs(self):
        return {"grid_path": self.grid_path,
                "grid_path_length": len(self.grid_path)}


class TopoMap(Graph):

    """To create a TopoMap,
    construct a mapping called e.g. `edges` that maps from edge id to Edge,
    and do TopoMap(edges)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_closest = {}
        self._cache_shortest_path = {}

    def closest_node(self, x, y):
        """Given a point at (x,y) find the node that is closest to this point.
        """
        if (x,y) in self._cache_closest:
            return self._cache_closest[(x,y)]
        else:
            nid = min(self.nodes,
                       key=lambda nid: euclidean_dist(self.nodes[nid].pos, (x,y)))
            self._cache_closest[(x,y)] = nid
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
