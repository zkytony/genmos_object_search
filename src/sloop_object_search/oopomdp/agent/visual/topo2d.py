from .basic2d import VizSloopMosBasic2D
from sloop_object_search.oopomdp.models.topo_map import draw_topo as draw_topo_func

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
        return super().render(agent, objlocs, colors={},
                              robot_state=robot_state, draw_fov=draw_fov,
                              draw_belief=draw_belief, img=img)
