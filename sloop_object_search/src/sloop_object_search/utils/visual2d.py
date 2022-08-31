"""image-based 2D visualization of beliefs, grid maps, etc."""
import math
import random
import cv2
import numpy as np
import pygame
from tqdm import tqdm

from .images import overlay, cv2shape
from .colors import lighter, lighter_with_alpha, inverse_color_rgb, random_unique_color
from .math import to_rad, to_deg
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2

__all__ = ['Visualizer2D']


########### Parent class ############
class Visualizer2D:

    def __init__(self, **config):
        """
        2D visualizer using pygame.

        config entries:
        - res: resolution
        - region: an object with properties width, length, obstacles
        - linewidth: line width when drawing grid cells
        - bg_path: Path to an image to place in the background
        - colors: maps from object id to (r, g, b)
        """
        self._res = config.get("res", 30)   # resolution
        self._region = config.get("region", None)
        self._linewidth = config.get("linewidth", 1)
        self._bg_path = config.get("bg_path", None)
        self._bg_img = config.get("bg_img", None)
        self._colors = config.get("colors", {})
        self._obstacle_color = config.get("obstacle_color", (40, 3, 10))
        self._unknown_color = config.get("unknown_color", (168, 168, 168))
        self._initialized = False
        self._rnd = random.Random(100) # sudo random for generating color

    @property
    def img_width(self):
        return self._region.width * self._res

    @property
    def img_height(self):
        return self._region.length * self._res

    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._initialized = True

    def on_cleanup(self):
        pygame.display.quit()
        pygame.quit()

    def set_bg(self, bgimg):
        self._bg_img = bgimg

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._region.width, self._region.length
        img = np.full((w*r, l*r, 4), 255, dtype=np.uint8)

        # Make an image of grids
        bgimg = None
        if self._bg_img is not None:
            bgimg = self._bg_img
        elif self._bg_path is not None:
            bgimg = cv2.imread(self._bg_path, cv2.IMREAD_UNCHANGED)
            bgimg = cv2.cvtColor(bgimg, cv2.COLOR_BGR2RGB)
            bgimg = cv2.flip(bgimg, 1)  # flip horizontally
            bgimg = cv2.rotate(bgimg, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise

        if bgimg is not None:
            bgimg = cv2.resize(bgimg, (w*r, l*r))
            img = overlay(img, bgimg, opacity=1.0)

        for x in range(w):
            for y in range(l):
                if self._obstacle_color is not None:
                    if (x, y) in self._region.obstacles:
                        cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                      self._obstacle_color, -1)

                if self._unknown_color is not None:
                    if hasattr(self._region, "unknown") and (x, y) in self._region.unknown:
                        cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                      self._unknown_color, -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), self._linewidth)
        return img

    def visualize(self, *args, **kwargs):
        print("Deprication warning: visualize is depreciated. Use render followed show_img instead")
        return self.show_img(self.render(*args, **kwargs))

    def render(self):
        return self._make_gridworld_image(self._res)

    def highlight(self, img, locations, color=(128,128,128),
                  shape="rectangle", alpha=1.0, show_progress=False, scale=1.0):
        r = self._res
        for loc in tqdm(locations, disable=not show_progress):
            x, y = loc
            if shape == 'rectangle':
                shift = (r - scale*r) / 2
                topleft = (int(round(y*r + shift)),
                           int(round(x*r + shift)))
                bottomright = (int(round(y*r + r - shift)),
                               int(round(x*r + r - shift)))
                img = cv2shape(img, cv2.rectangle,
                               topleft, bottomright,
                               color, -1, alpha=alpha)
            elif shape == 'circle':
                size = scale*r
                radius = int(round(size / 2))
                shift = int(round(r / 2))
                img = cv2shape(img, cv2.circle,
                               (y*r+shift, x*r+shift),
                               radius, color, -1, alpha=alpha)
            else:
                raise ValueError(f"Unknown shape {shape}")
        return img

    def show_img(self, img, rotation=None, flip_horizontally=False, flip_vertically=False):
        """
        Internally, the img origin (0,0) is top-left (that is the opencv image),
        so +x is right, +y is down.
        But when displaying, the image is flipped, so that in the displayed image, +x is right, +y is up.

        rotation: a cv2 constant (e.g. cv2.ROTATE_90_CLOCKWISE), which
        will rotate the image before it is drawn.

        flip is applied after rotation.
        """
        if not self._initialized:
            self.on_init()
            self._initialized = True
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if rotation is not None:
            img = cv2.rotate(img, rotation)
        if flip_horizontally:
            img = cv2.flip(img, 1)
        if flip_vertically:
            img = cv2.flip(img, 0)
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()
        return img

    def get_color(self, objid, colors=None, alpha=1.0):
        """
        colors: maps frmo objid to [r,g,b]. If None, then the
            self._colors will be used instead. If objid not in
            colors, then a pseudo-random color will be generated.
        """
        if colors is None:
            colors = self._colors
        if objid not in colors:
            color = random_unique_color(self._colors.values(), rnd=self._rnd, fmt='rgb')
            colors[objid] = color
        else:
            color = colors[objid]
        if len(color) == 3 and alpha is not None:
            color = color + [int(round(alpha*255))]
        color = tuple(color)
        return color

    ### Functions to draw
    def draw_robot(self, img, x, y, th, color=(255, 150, 0), thickness=2):
        """Note: agent by default (0 angle) looks in the +z direction in Unity,
        which corresponds to +y here. That's why I'm multiplying y with cos."""
        size = self._res
        x *= self._res
        y *= self._res

        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=thickness)

        if th is not None:
            endpoint = (y+shift + int(round(shift*math.sin(to_rad(th)))),
                        x+shift + int(round(shift*math.cos(to_rad(th)))))
            cv2.line(img, (y+shift,x+shift), endpoint, color, 2)
        return img

    def draw_object_belief(self, img, belief, color,
                           circle_drawn=None, shape="circle"):
        """
        circle_drawn: map from pose to number of times drawn;
            Used to determine size of circle to draw at a location
        """
        if circle_drawn is None:
            circle_drawn = {}
        size = self._res * 0.85
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        last_val = -1
        hist = belief.get_histogram()
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = lighter_with_alpha(color, 1-hist[state]/last_val)

            if len(color) == 4:
                stop = color[3]/255 < 0.1
            else:
                stop = np.mean(np.array(color[:3]) / np.array([255, 255, 255])) < 0.999

            if not stop:
                tx, ty = self._get_viz_pos(state.loc)
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                if shape == "rectangle":
                    img = cv2shape(img, cv2.rectangle,
                                   (ty*self._res,
                                    tx*self._res),
                                   (ty*self._res+self._res,
                                    tx*self._res+self._res),
                                   color, thickness=-1, alpha=color[3]/255)
                elif shape == "circle":
                    img = cv2shape(img, cv2.circle, (ty*self._res + shift,
                                                     tx*self._res + shift), radius, color,
                                   thickness=-1, alpha=color[3]/255)
                else:
                    raise ValueError(f"Unknown shape {shape}")
                last_val = hist[state]
                if last_val <= 0:
                    break
        return img

    def _get_viz_pos(self, pos):
        """Given a position, return the position
        on the canvas that should be used for visualization
        at that position. Used to minimize code duplication
        in child classes"""
        return pos


########### GridMap visualizer ###############
class GridMapVisualizer(Visualizer2D):
    """
    Visualizer for a given grid map (GridMap).
    """
    def __init__(self, **config):
        """
        Visualizer for grid map (GridMap).

        config entries:
            grid_map: GridMap
        """
        self._grid_map = config.get("grid_map", None)
        super().__init__(**config)
        self._region = self._grid_map

    def render(self):
        return self._make_gridworld_image(self._res)

    def draw_fov(self, img, sensor, robot_pose, color=[233, 233, 8]):
        size = self._res // 2
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        for x in range(self._region.width):
            for y in range(self._region.length):
                if sensor.in_range((x,y), robot_pose):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.7)
        return img


########### GridMap2 visualizer ###############
class GridMap2Visualizer(Visualizer2D):
    """
    Visualizer for a given grid map (GridMap2).
    """
    def __init__(self, **config):
        self._grid_map2 = config.get("grid_map", None)
        assert isinstance(self._grid_map2, GridMap2),\
            "GridMap2Visualizer expects grid_map2 to be GridMap2."
        # This is what will be used for plotting by Visualizer2D
        self._grid_map = self._grid_map2.to_grid_map()
        super().__init__(**config)
        self._region = self._grid_map

    def render(self):
        return self._make_gridworld_image(self._res)

    def highlight(self, img, locations, color=(128,128,128),
                  shape="rectangle", alpha=1.0, show_progress=False, scale=1.0):
        """
        'locations' here should be locations on GridMap2. Thus
        when visualizing, we need to shift them to nonnegative coordinates."""
        shifted_locations = [self._grid_map2.shift_pos(*loc)
                             for loc in locations]
        return super().highlight(img, shifted_locations, color=color,
                                 shape=shape, alpha=alpha, show_progress=show_progress,
                                 scale=scale)

    def draw_robot(self, img, x, y, th, color=(255, 150, 0), thickness=2):
        shifted_x, shifted_y = self._grid_map2.shift_pos(x, y)
        return super().draw_robot(img, shifted_x, shifted_y, th, color=(255, 150, 0), thickness=2)

    def _get_viz_pos(self, pos):
        return self._grid_map2.shift_pos(*pos)

    def draw_fov(self, img, sensor, robot_pose, color=[233, 233, 8]):
        size = self._res // 2
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        for x in range(self._region.width):
            for y in range(self._region.length):
                # (x,y) here is not a GridMap2 location; (it's shifted to
                # be nonnegative); We need to shift it back.
                if sensor.in_range(
                        self._grid_map2.shift_back_pos(x,y), robot_pose):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.7)
        return img


########### VizSloopMos*Basic2D* visualizer ###############
class VizSloopMosBasic2D(GridMap2Visualizer):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)

    def render(self, agent, objlocs={}, colors={}, robot_state=None, draw_fov=None,
               draw_belief=True, img=None, **kwargs):
        """
        Args:
            agent (CosAgent)
            robot_state (RobotState): 2d robot state
            target_belief (Histogram) target belief
            objlocs (dict): maps from object id to true object (x,y) location tuple
            colors (dict): maps from objid to [R,G,B]
            draw_fov (list): draw the FOV for the detectors for this objects
        """
        if robot_state is None:
            robot_state = agent.belief.mpe().s(agent.robot_id)

        assert robot_state.is_2d, "2D visualizer expects 2D robot state"

        if img is None:
            img = self._make_gridworld_image(self._res)
        x, y, th = robot_state["pose"]
        for objid in sorted(objlocs):
            img = self.highlight(img,
                                 [objlocs[objid]],
                                 color=self.get_color(objid, colors, alpha=None))
        if draw_belief:
            img = self.draw_object_beliefs(agent.beleif.object_beliefs)

        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        if draw_fov is not None:
            for objid in sorted(draw_fov):
                img = VizSloopMosBasic2D.draw_fov(
                    self, img, agent.sensor(objid),
                    robot_state,
                    inverse_color_rgb(self.get_color(
                        objid, colors, alpha=None)))
        return img

    def draw_object_beliefs(self, img, object_beliefs):
        for objid in object_beliefs:
            color = self.get_color(objid, self._colors, alpha=None)
            belief_obj = object_beliefs[objid]
            img = self.draw_object_belief(img, belief_obj,
                                          list(color) + [250])
        return img


########### VizSloopMos*Topo* visualizer ###############
class VizSloopMosTopo(VizSloopMosBasic2D):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)
        self._draw_topo_grid_path = config.get("draw_topo_grid_path", False)
        self._mark_cell_kwargs = config.get("topo_mark_cell", {})

    def render(self, topo_map=None, object_beliefs=None,
               robot_id=None, robot_pose=None, img=None):
        if img is None:
            img = self._make_gridworld_image(self._res)
        # if topo_map is not None:
        #     img = draw_topo_func(img, topo_map, self._res,
        #                          draw_grid_path=self._draw_topo_grid_path,
        #                          **self._mark_cell_kwargs)
        if object_beliefs is not None:
            img = self.draw_object_beliefs(img, object_beliefs)
        if robot_pose is not None:
            img = self.draw_robot(img, *robot_pose,
                                  color=self.get_color(robot_id, self._colors, alpha=None))
        return img

    # def render_old(self, agent, objlocs={}, colors={},
    #                robot_state=None, draw_fov=None,
    #                draw_belief=True, img=None, draw_topo=True, **mark_cell_kwargs):
    #     """render image"""
    #     if img is None:
    #         img = self._make_gridworld_image(self._res)

    #     img = super().render(agent, objlocs=objlocs, colors=colors,
    #                           robot_state=robot_state, draw_fov=draw_fov,
    #                           draw_belief=draw_belief, img=img)

    #     # Draw topo map
    #     if draw_topo:
    #         img = draw_topo_func(img, agent.topo_map, self._res,
    #                              draw_grid_path=self._draw_topo_grid_path,
    #                              **mark_cell_kwargs)

    #     # redraw robot on top of topo map
    #     if robot_state is None:
    #         robot_state = agent.belief.mpe().s(agent.robot_id)
    #     x, y, th = robot_state["pose"]
    #     img = self.draw_robot(img, x, y, th, (255, 20, 20))
    #     return img


#------ Visualization for topo map -----#
# In all fucntions, r means resolution, in pygmae visualziation
def draw_edge(img, pos1, pos2, r, thickness=2, color=(0, 0, 0)):
    x1, y1 = pos1
    x2, y2 = pos2
    cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
             color, thickness=thickness)
    return img

def draw_topo_func(img, topo_map, r, draw_grid_path=False, path_color=(52, 235, 222),
                   edge_color=(200, 40, 20), edge_thickness=2, linewidth=2,
                   **mark_cell_kwargs):
    """
    Draws topological map on the image `img`.

    linewidth: the linewidth of the bounding box when drawing grid path
    edge_thickness: the thickness of the edge on the topo map.

    flip: flip the text horizontally (you might need to set this False if you
    flip the image horizontally already when calling show_img)
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
        img = mark_cell(img, pos, int(nid), r, **mark_cell_kwargs)
    return img

def mark_cell(img, pos, nid, r, linewidth=1, unmark=False,
              show_img_flip_horizontally=False):
    """show_img_flip_horizontally: True if show_img flips image horizontally; """
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
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if show_img_flip_horizontally:
            # do this if show_img has flip_horizontally=True
            imgtxt = cv2.flip(imgtxt, 0)
            imgtxt = cv2.flip(imgtxt, 1)
        else:
            # do this if show_img has flip_horizontally=False
            imgtxt = cv2.flip(imgtxt, 1) # flip horizontally
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
    return img
