import time
from sloop.utils.visual import GridMapVisualizer
from sloop.grid_map import GridMap

def test_gridmap_visuailzer():
    grid_map = GridMap(5, 5,
                       {(2,3), (4,0), (4,1), (4,2)},
                       unknown={(3,3), (3,4)})
    viz = GridMapVisualizer(grid_map=grid_map)
    img = viz.render()
    viz.show_img(img)
    time.sleep(5)
