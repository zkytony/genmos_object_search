import cv2
from genmos_object_search.utils.visual2d import GridMapVisualizer
from genmos_object_search.utils.colors import inverse_color_rgb
from genmos_object_search.utils.images import cv2shape
from genmos_object_search.utils.math import to_deg


class VizSloopMosBasic2D(GridMapVisualizer):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)

    def draw_fov(self, img, sensor, robot_state,
                 color=[233, 233, 8]):
        # We will draw what's in mean range differently from the max range.
        size = self._res // 2
        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        for x in range(self._region.width):
            for y in range(self._region.length):
                if hasattr(self._region, "unknown") and (x, y) in self._region.unknown:
                    continue  # occluded (don't draw; the model doesn't care about this though but it is ok for now)

                if robot_state.loc_in_range(sensor, (x,y), use_mean=False):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.4)

                if robot_state.loc_in_range(sensor, (x,y), use_mean=True):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+shift, x*self._res+shift),
                                   radius, color, thickness=-1, alpha=0.7)
        return img


    def render(self, agent, objlocs, colors={}, robot_state=None, draw_fov=None,
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
            for objid in agent.belief.object_beliefs:
                color = self.get_color(objid, colors, alpha=None)
                belief_obj = agent.belief.b(objid)
                img = self.draw_object_belief(img, belief_obj,
                                              list(color) + [250])
        img = self.draw_robot(img, x, y, th, (255, 20, 20))
        if draw_fov is not None:
            for objid in sorted(draw_fov):
                img = VizSloopMosBasic2D.draw_fov(
                    self, img, agent.sensor(objid),
                    robot_state,
                    inverse_color_rgb(self.get_color(
                        objid, colors, alpha=None)))
        return img
