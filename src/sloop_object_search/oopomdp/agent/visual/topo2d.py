from .basic2d import VizSloopMosBasic2D

class VizSloopMosTopo(VizSloopMosBasic2D):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)

    def render(self, agent, objlocs, colors={},
               robot_state=None, draw_fov=None,
               draw_belief=True, img=None):
        """render image"""
        if img is None:
            img = self._make_gridworld_image(self._res)

        # Draw topo map
        img = draw_topo(img, agent.topo_map, self._res,
                        draw_grid_path=self._draw_topo_grid_path)
        return super().render(agent, objlocs, colors={},
                              robot_state=robot_State, draw_fov=draw_fov,
                              draw_belief=draw_belief, img=img)
