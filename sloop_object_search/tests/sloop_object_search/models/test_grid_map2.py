import time
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.utils.visual2d import GridMap2Visualizer
from sloop_object_search.utils import grid_map_utils

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

def test_grid_map2_with_real_map():
    obstacles = {(26, 21), (18, 17), (7, 26), (18, 26), (19, 9), (30, 9), (42, 11), (11, 5), (30, 18), (11, 23), (41, 24), (33, 20), (22, 10), (34, 12), (3, 6), (37, 8), (22, 19), (3, 15), (3, 24), (14, 24), (15, 7), (38, 9), (7, 3), (18, 3), (38, 18), (18, 12), (15, 25), (18, 21), (8, 4), (19, 4), (30, 4), (30, 13), (41, 10), (33, 15), (10, 31), (22, 5), (22, 14), (14, 10), (3, 10), (22, 23), (37, 12), (37, 21), (3, 19), (3, 28), (38, 4), (15, 11), (18, 7), (7, 7), (18, 16), (29, 13), (21, 18), (11, 4), (41, 5), (33, 10), (41, 23), (2, 13), (10, 26), (2, 22), (22, 9), (34, 11), (14, 5), (3, 5), (22, 18), (37, 7), (3, 14), (22, 27), (3, 23), (14, 23), (14, 32), (15, 6), (17, 25), (18, 11), (40, 8), (29, 8), (29, 17), (40, 17), (21, 13), (6, 24), (21, 22), (41, 9), (33, 5), (10, 3), (33, 14), (2, 8), (2, 17), (2, 26), (22, 4), (34, 6), (22, 13), (3, 9), (14, 9), (37, 11), (3, 18), (17, 20), (9, 25), (40, 12), (29, 12), (21, 8), (40, 21), (21, 17), (21, 26), (41, 13), (33, 9), (2, 3), (33, 18), (2, 12), (10, 25), (2, 21), (22, 8), (3, 4), (14, 4), (13, 36), (36, 10), (17, 6), (36, 19), (28, 15), (17, 24), (9, 20), (17, 33), (6, 5), (29, 7), (29, 16), (40, 16), (21, 12), (6, 23), (40, 25), (21, 21), (21, 30), (33, 4), (2, 7), (32, 8), (2, 16), (32, 17), (24, 13), (35, 22), (36, 14), (5, 8), (17, 10), (9, 6), (28, 19), (17, 19), (5, 26), (9, 24), (29, 11), (40, 11), (21, 7), (21, 16), (20, 20), (10, 6), (12, 25), (2, 11), (32, 12), (35, 8), (32, 21), (1, 15), (35, 17), (16, 13), (1, 24), (16, 31), (36, 9), (5, 3), (17, 5), (36, 18), (28, 14), (17, 23), (6, 4), (42, 25), (29, 6), (19, 32), (39, 10), (39, 19), (31, 15), (20, 15), (20, 24), (4, 25), (32, 7), (13, 3), (32, 16), (43, 16), (24, 12), (35, 12), (16, 8), (1, 19), (35, 21), (24, 21), (16, 26), (36, 4), (36, 13), (5, 7), (17, 9), (28, 18), (5, 25), (42, 20), (39, 5), (39, 14), (11, 32), (20, 10), (31, 10), (12, 6), (20, 19), (31, 19), (54, 21), (23, 15), (12, 24), (32, 11), (13, 7), (35, 7), (32, 20), (1, 14), (35, 16), (16, 12), (1, 23), (17, 4), (15, 34), (42, 6), (19, 13), (42, 15), (19, 22), (42, 24), (19, 31), (39, 9), (31, 5), (20, 5), (39, 18), (20, 14), (31, 14), (20, 23), (4, 6), (23, 19), (4, 24), (32, 6), (13, 2), (35, 11), (16, 7), (1, 18), (38, 13), (38, 22), (5, 6), (7, 25), (18, 25), (19, 8), (30, 8), (27, 21), (42, 10), (30, 17), (19, 17), (42, 19), (8, 26), (19, 26), (0, 22), (11, 22), (39, 4), (39, 13), (20, 9), (31, 9), (39, 22), (12, 5), (31, 18), (20, 18), (23, 14), (12, 23), (34, 20), (53, 33), (1, 4), (35, 6), (38, 8), (26, 15), (15, 24), (18, 20), (15, 33), (8, 3), (19, 3), (42, 5), (19, 12), (42, 14), (30, 12), (19, 21), (42, 23), (0, 17), (11, 26), (39, 8), (20, 4), (31, 4), (12, 0), (20, 13), (31, 13), (23, 9), (4, 5), (23, 18), (22, 22), (37, 20), (3, 27), (14, 36), (15, 10), (38, 12), (7, 6), (18, 6), (38, 21), (18, 15), (7, 24), (18, 24), (18, 33), (30, 7), (19, 7), (42, 9), (11, 3), (30, 16), (19, 25), (11, 21), (41, 22), (12, 4), (34, 10), (37, 6), (22, 17), (34, 19), (53, 32), (3, 13), (14, 13), (22, 26), (3, 22), (15, 5), (38, 7), (26, 14), (18, 10), (18, 19), (15, 32), (42, 4), (19, 11), (30, 11), (11, 7), (41, 8), (0, 16), (33, 13), (10, 29), (2, 25), (34, 5), (22, 12), (34, 14), (3, 8), (14, 8), (22, 21), (3, 17), (22, 30), (3, 26), (15, 9), (26, 9), (38, 11), (7, 5), (18, 5), (18, 14), (7, 23), (40, 20), (30, 6), (21, 25), (19, 6), (11, 2), (41, 12), (33, 8), (41, 21), (33, 17), (25, 13), (10, 24), (2, 20), (25, 22), (22, 7), (34, 9), (3, 3), (14, 3), (22, 16), (34, 18), (37, 5), (3, 12), (14, 12), (22, 25), (3, 21), (37, 23), (15, 4), (15, 13), (18, 9), (18, 18), (29, 15), (21, 11), (6, 22), (21, 20), (41, 7), (33, 12), (41, 25), (2, 6), (33, 21), (2, 15), (2, 24), (34, 4), (22, 11), (3, 7), (14, 7), (22, 20), (37, 9), (3, 16), (14, 25), (3, 25), (17, 27), (18, 4), (7, 4), (6, 8), (40, 10), (29, 10), (21, 6), (29, 19), (21, 15), (6, 26), (21, 24), (41, 11), (33, 7), (41, 20), (33, 16), (2, 10), (2, 19), (10, 32), (22, 6), (13, 25), (14, 2), (3, 2), (37, 4), (14, 11), (3, 11), (36, 8), (36, 17), (17, 13), (28, 13), (17, 31), (6, 3), (29, 5), (40, 14), (29, 14), (21, 10), (21, 19), (41, 6), (33, 11), (2, 5), (2, 14), (25, 16), (32, 15), (2, 23), (35, 20), (14, 6), (16, 25), (36, 12), (17, 8), (9, 4), (28, 17), (5, 24), (17, 26), (6, 7), (29, 9), (40, 9), (21, 5), (29, 18), (40, 18), (21, 14), (6, 25), (21, 23), (10, 4), (2, 9), (32, 10), (13, 6), (2, 18), (32, 19), (35, 15), (16, 11), (13, 24), (1, 22), (36, 7), (17, 3), (17, 12), (9, 8), (9, 26), (29, 4), (40, 4), (40, 13), (21, 9), (39, 17), (20, 22), (2, 4), (4, 23), (32, 5), (32, 14), (35, 10), (16, 6), (1, 17), (35, 19), (1, 26), (16, 24), (16, 33), (36, 11), (5, 5), (17, 7), (36, 20), (9, 3), (28, 16), (5, 23), (9, 21), (42, 18), (6, 6), (8, 25), (21, 4), (39, 12), (20, 8), (31, 8), (39, 21), (31, 17), (20, 17), (23, 13), (20, 26), (13, 5), (35, 5), (32, 18), (35, 14), (1, 12), (16, 10), (1, 21), (13, 23), (17, 11), (5, 9), (28, 11), (9, 7), (27, 15), (42, 13), (19, 20), (30, 20), (11, 25), (39, 7), (39, 16), (31, 12), (20, 12), (20, 21), (4, 4), (54, 23), (23, 17), (20, 30), (4, 13), (12, 26), (15, 12), (32, 4), (4, 22), (32, 13), (35, 9), (16, 5), (1, 16), (35, 18), (1, 25), (38, 20), (5, 4), (15, 36), (8, 6), (42, 8), (42, 17), (19, 15), (30, 15), (0, 11), (19, 24), (8, 24), (11, 29), (39, 11), (31, 7), (20, 7), (39, 20), (12, 3), (20, 16), (31, 16), (23, 12), (20, 25), (4, 8), (23, 21), (4, 17), (4, 26), (13, 4), (35, 4), (1, 11), (35, 13), (16, 9), (1, 20), (38, 6), (26, 22), (15, 31), (19, 10), (30, 10), (42, 12), (11, 6), (19, 19), (30, 19), (11, 24), (39, 6), (0, 24), (39, 15), (31, 11), (20, 11), (12, 7), (31, 20), (4, 3), (34, 13), (4, 21), (16, 4), (14, 34), (15, 8), (38, 10), (38, 19), (18, 13), (15, 26), (27, 9), (19, 5), (30, 5), (42, 7), (11, 1), (19, 14), (42, 16), (30, 14), (19, 23), (0, 19), (20, 6), (31, 6), (4, 7), (34, 8), (22, 15), (34, 17), (37, 13), (22, 24), (3, 20), (37, 22), (38, 14), (18, 8)}
    free_locations = {(15, 21), (6, 18), (16, 20), (7, 17), (4, 9), (5, 10), (8, 9), (14, 22), (5, 19), (8, 18), (19, 18), (17, 21), (11, 14), (9, 17), (13, 8), (13, 17), (15, 14), (6, 11), (7, 10), (15, 23), (6, 20), (16, 22), (7, 19), (12, 18), (14, 15), (17, 14), (5, 12), (8, 11), (9, 10), (5, 21), (9, 19), (11, 16), (13, 10), (10, 20), (13, 19), (15, 16), (6, 13), (16, 15), (7, 12), (7, 21), (12, 20), (14, 17), (17, 16), (5, 14), (9, 12), (11, 9), (13, 12), (10, 22), (13, 21), (15, 18), (16, 17), (7, 14), (18, 23), (12, 13), (12, 22), (14, 19), (5, 16), (4, 18), (9, 14), (10, 15), (13, 14), (16, 19), (7, 16), (12, 15), (4, 11), (14, 21), (5, 18), (4, 20), (9, 16), (10, 8), (8, 20), (10, 17), (13, 16), (7, 9), (16, 21), (12, 8), (12, 17), (14, 14), (5, 11), (9, 9), (8, 13), (10, 10), (13, 9), (8, 22), (10, 19), (13, 18), (11, 18), (16, 14), (7, 11), (16, 23), (6, 15), (12, 10), (12, 19), (14, 16), (5, 13), (4, 15), (17, 18), (8, 15), (10, 12), (13, 11), (11, 11), (10, 21), (9, 23), (11, 20), (16, 16), (15, 20), (6, 17), (12, 12), (12, 21), (14, 18), (8, 8), (8, 17), (10, 14), (11, 13), (10, 23), (6, 10), (15, 22), (6, 19), (7, 18), (12, 14), (4, 10), (4, 19), (8, 10), (10, 7), (5, 20), (17, 22), (10, 16), (9, 18), (8, 19), (11, 15), (15, 15), (6, 12), (6, 21), (7, 20), (12, 16), (4, 12), (17, 15), (8, 12), (10, 9), (9, 11), (11, 8), (5, 22), (8, 21), (10, 18), (11, 17), (13, 20), (15, 17), (6, 14), (7, 13), (12, 9), (18, 22), (7, 22), (4, 14), (5, 15), (17, 17), (10, 11), (9, 13), (8, 14), (11, 10), (8, 23), (9, 22), (11, 19), (13, 13), (13, 22), (15, 19), (6, 16), (16, 18), (7, 15), (12, 11), (4, 16), (14, 20), (5, 17), (19, 16), (8, 16), (9, 15), (11, 12), (10, 13), (13, 15), (6, 9), (7, 8)}
    grid_map2 = GridMap2(obstacles=obstacles,
                        free_locations=free_locations)
    # cells_with_minimum_distance_from_obstacles
    cells = grid_map_utils.cells_with_minimum_distance_from_obstacles(grid_map2, dist=1)
    viz = GridMap2Visualizer(grid_map=grid_map2, res=30)
    img = viz.render()
    img = viz.highlight(img, cells, color=(72, 213, 235))
    viz.show_img(img)
    time.sleep(3)

    # obstacles around free locations
    cells = grid_map_utils.obstacles_around_free_locations(grid_map2, dist=1)
    img = viz.render()
    img = viz.highlight(img, cells, color=(72, 213, 235))
    viz.show_img(img)
    time.sleep(3)

if __name__ == "__main__":
    # test_grid_map2()
    test_grid_map2_with_real_map()
