import cv2
import numpy as np
from .basic2d import VizSloopMosBasic2D

class VizSloopMosTopo(VizSloopMosBasic2D):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)
        self._draw_topo_grid_path = config.get("draw_topo_grid_path", False)

    def render(self, agent, objlocs, colors={},
               robot_state=None, draw_fov=None,
               draw_belief=True, img=None, draw_topo=True):
        """render image"""
        if img is None:
            img = self._make_gridworld_image(self._res)

        # Draw topo map
        if draw_topo:
            img = draw_topo_func(img, agent.topo_map, self._res,
                                 draw_grid_path=self._draw_topo_grid_path)
        return super().render(agent, objlocs, colors=colors,
                              robot_state=robot_state, draw_fov=draw_fov,
                              draw_belief=draw_belief, img=img)


#------ Visualization -----#
# In all fucntions, r means resolution, in pygmae visualziation
def draw_edge(img, pos1, pos2, r, thickness=2, color=(0, 0, 0)):
    x1, y1 = pos1
    x2, y2 = pos2
    cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
             color, thickness=thickness)
    return img

def draw_topo_func(img, topo_map, r, draw_grid_path=False, path_color=(52, 235, 222),
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
