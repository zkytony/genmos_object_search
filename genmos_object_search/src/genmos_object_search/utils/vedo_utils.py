import vedo
import numpy as np
from genmos_object_search.utils.colors import color_map, cmaps
from genmos_object_search.utils import math as math_utils
from genmos_object_search.oopomdp.models.octree_belief import RegionalOctreeDistribution, Octree
from genmos_object_search.oopomdp.models.search_region import SearchRegion3D, SearchRegion2D


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



def draw_octree_dist(octree_dist, viz=True, color_by_prob=True,
                     cmap=cmaps.COLOR_MAP_GRAYS):
    """draw the octree dist in POMDP space."""
    actors = []
    # visualize octree
    voxels = octree_dist.collect_plotting_voxels()
    vp = [v[:3] for v in voxels]
    vr = [v[3] for v in voxels]  # resolutions
    probs = [octree_dist.prob_at(
        *Octree.increase_res(v[:3], 1, v[3]), v[3])
             for v in voxels]
    for i in range(len(vp)):
        if color_by_prob:
            color = color_map(probs[i], [min(probs), max(probs)], cmap)
            alpha = math_utils.remap(probs[i], min(probs)-0.00001, max(probs), 0.001, 0.8)
        else:
            color = [0.9, 0.1, 0.1]
            alpha = 0.6
        pos = vp[i]
        size = vr[i]  # cube size in meters
        cube = vedo.Cube(pos=pos, side=size, c=color, alpha=alpha)
        actors.append(cube)
    if viz:
        vedo.show(actors).close()
    return actors
