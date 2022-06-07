import cv2
import numpy as np
from collections import deque
import networkx as nx
from sloop_object_search.utils.color import lighter
from sloop_object_search.utils.math import euclidean_dist
from sloop_object_search.utils.graph import Node, Graph, Edge

class TopoNode(Node):
    """TopoNode is a node on the grid map."""

    def __init__(self, node_id, grid_pos, search_region_locs):
        """
        grid_pos (x,y): The grid position this node is located
        search_region_locs (list or set): locations where the
            target object can be for which this node is the closest.
        """
        super().__init__(node_id, grid_pos)
        self._coords = grid_pos
        self.search_region_locs = search_region_locs

    @property
    def pos(self):
        return self.data

    def prob(self, target_hist):
        return sum(target_hist[loc]
                   for loc in self.search_region_locs)

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

    def total_prob(self, target_hist):
        return sum(self.nodes[nid].prob(target_hist)
                   for nid in self.nodes)


#------ Visualization -----#
# In all fucntions, r means resolution, in pygmae visualziation
def draw_edge(img, pos1, pos2, r, thickness=2, color=(0, 0, 0)):
    x1, y1 = pos1
    x2, y2 = pos2
    cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
             color, thickness=thickness)
    return img

def draw_topo(img, topo_map, r, draw_grid_path=False, path_color=(52, 235, 222),
              edge_color=(200, 40, 20), edge_thickness=2, linewidth=2):
    """
    Draws topological map on the image `img`.

    linewidth: the linewidth of the bounding box when drawing grid path
    edge_thickness: the thickness of the edge on the topo map.
    """
    for eid in topo_map.edges:
        edge = topo_map.edges[eid]
        if draw_grid_path:
            if edge.grid_path is not None:
                for x, y in edge.grid_path:
                    cv2.rectangle(img,
                                  (y*r, x*r),
                                  (y*r+r, x*r+r),
                                  path_color, -1)
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  lighter(path_color, 0.7), linewidth)

    for eid in topo_map.edges:
        edge = topo_map.edges[eid]
        if not edge.degenerate:
            node1, node2 = edge.nodes
            pos1 = node1.pos
            pos2 = node2.pos
            img = draw_edge(img, pos1, pos2, r, edge_thickness, color=edge_color)

    for nid in topo_map.nodes:
        pos = topo_map.nodes[nid].pos
        img = mark_cell(img, pos, int(nid), r)
    return img

def mark_cell(img, pos, nid, r, linewidth=1, unmark=False):
    if unmark:
        color = (255, 255, 255, 255)
    else:
        color = (242, 227, 15, 255)
    x, y = pos
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  color, -1)
    # Draw boundary
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  (0, 0, 0), linewidth)

    if not unmark:
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.72
        fontColor              = (43, 13, 4)
        lineType               = 1
        imgtxt = np.full((r, r, 4), color, dtype=np.uint8)
        text_loc = (int(round(r/4)), int(round(r/1.5)))
        cv2.putText(imgtxt, str(nid), text_loc, #(y*r+r//4, x*r+r//2),
                    font, fontScale, fontColor, lineType)
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_CLOCKWISE)
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
    return img
