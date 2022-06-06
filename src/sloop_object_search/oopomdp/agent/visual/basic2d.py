from sloop_object_search.utils.visual import GridMapVisualizer

class VizSloopMosBasic2D(GridMapVisualizer):
    def __init__(self, grid_map, **config):
        super().__init__(grid_map=grid_map, **config)
