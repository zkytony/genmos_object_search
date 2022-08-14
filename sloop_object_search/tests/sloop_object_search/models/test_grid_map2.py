import time
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.utils.visual import GridMap2Visualizer

def test_grid_map2():
    free_locations = {(x,y)
                      for x in range(2,8)
                      for y in range(2,6)}
    obstacles = {(x,y)
                 for x in range(-5,1)
                 for y in range(-1,2)}
    grid_map2 = GridMap2(obstacles=obstacles, free_locations=free_locations)
    viz = GridMap2Visualizer(grid_map=grid_map2, res=30)
    img = viz.render()
    viz.show_img(img)
    time.sleep(5)

if __name__ == "__main__":
    test_grid_map2()
