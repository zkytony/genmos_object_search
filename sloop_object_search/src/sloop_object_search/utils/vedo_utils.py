import numpy as np
import vedo

def draw_search_region2d(search_region, grid_robot_position=None, points=None):
    actors = []
    if points is not None:
        map_vpts = vedo.Points(points, c=(0.6, 0.6, 0.6))
        actors.append(map_vpts)

    grid_map = search_region.grid_map

    freeloc_points = np.asarray(list(search_region.to_world_pos(loc)
                                     for loc in grid_map.free_locations))
    freeloc_points = np.append(freeloc_points, np.zeros((len(freeloc_points), 1)), axis=1)
    freeloc_vpts = vedo.Points(freeloc_points, c=(0.8, 0.8, 0.8))
    actors.append(freeloc_vpts)

    obloc_points = np.asarray(list(search_region.to_world_pos(loc)
                                   for loc in grid_map.obstacles))
    obloc_points = np.append(obloc_points, np.zeros((len(obloc_points), 1)), axis=1)
    obloc_vpts = vedo.Points(obloc_points, c=(0.2, 0.2, 0.2))
    actors.append(obloc_vpts)

    if grid_robot_position is not None:
        robot_point = np.array([[*search_region.to_world_pos(grid_robot_position), 1]])
        actors.append(vedo.Points(robot_point, c=(0.0, 0.8, 0.0)))

    vedo.show(actors).close()
    return actors
