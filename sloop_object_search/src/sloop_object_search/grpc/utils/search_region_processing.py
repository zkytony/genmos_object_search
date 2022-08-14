import math
import numpy as np
import open3d as o3d
from collections import deque
from tqdm import tqdm

from .proto_utils import pointcloudproto_to_array
from sloop_object_search.utils.math import remap, in_region, euclidean_dist
from sloop_object_search.utils.visual import GridMapVisualizer
from sloop_object_search.utils.conversion import Frame, convert
from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2


def search_region_from_occupancy_grid():
    pass


########### 2D search region ##############
def search_region_2d_from_point_cloud(point_cloud, robot_position, **kwargs):
    """
    The points in the given point cloud should correspond to static
    obstacles in the environment. The extent of this point cloud forms
    the extent of the search region.

    The point_cloud is to be projected down to 2D; We assume
    there is a "floor" (or flat plane) in the environment. "floor_cut"
    in kwargs specifies the height below which the points are considered
    to be part of the floor.
    """
    points_array = pointcloudproto_to_array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    grid_map = pcd_to_grid_map_2d(pcd, robot_position, debug=True, **kwargs)
    print("grid map created!")

def flood_fill_2d(grid_points, seed_point, grid_brush_size=2, flood_radius=None):
    """
    Given a numpy array of points that are supposed to be on a grid map,
    and a "seed point", flood fill by adding more grid points that are
    in empty locations to the set of grid points, starting from the seed
    point. Will not modify 'grid_points' but returns a new array.

    Args:
        grid_points (np.ndarray): a (n,2) or (n,3) array
        seed_point (np.ndarray or tuple): dimension should match that of a grid point
        grid_brush_size (int): The length (number of grids) of a square brush
            which will be used to fill out the empty spaces.
        flood_radius (float): the maximum euclidean distance between a
            point in the flood and the seed point (in grid units)
    """
    def _neighbors(p, d=1):
        # this works with both 2D or 3D points
        return set((p[0] + dx, p[1] + dy, *p[2:])
                   for dx in range(-d, d+1)
                   for dy in range(-d, d+1)
                   if not (dx == 0 and dy == 0))

    seed_point = tuple(seed_point)
    if grid_points.shape[1] != len(seed_point):
        raise ValueError("grid points and seed point have different dimensions.")

    xmax, ymax = np.max(grid_points, axis=0)[:2]
    xmin, ymin = np.min(grid_points, axis=0)[:2]
    _ranges = ([xmin, xmax], [ymin, ymax])
    grid_points_set = set(map(tuple, grid_points))
    # BFS
    worklist = deque([seed_point])
    visited = set({seed_point})
    new_points = set()
    while len(worklist) > 0:
        point = worklist.popleft()
        # Imagine placing a square brush centered at the point.
        # We assume that the point always represents a valid free cell
        brush_points = _neighbors(point, d=max(1, grid_brush_size//2))
        new_points.update(brush_points)
        if not any(bp in grid_points_set for bp in brush_points):
            # This brush stroke fits; we will consider all brush points
            # as potential free cells - i.e. neighbors
            for neighbor_point in brush_points:
                if neighbor_point not in visited:
                    if flood_radius is not None:
                        if euclidean_dist(neighbor_point, seed_point) > flood_radius:
                            continue  # skip this point: too far.
                    if in_region(neighbor_point[:2], _ranges):
                        worklist.append(neighbor_point)
                        visited.add(neighbor_point)
    return np.array(list(grid_points_set | new_points))


def pcd_to_grid_map_2d(pcd, robot_position, existing_map=None, **kwargs):
    """
    Given an Open3D point cloud object, the robot current pose, and
    optionally an existing grid map, output a GridMap2 object
    as the 2D projection of the point cloud.

    The algorithm works by first treating points above a certain
    height threshold (layout_cut) as points that form obstacles that
    identify the layout of the map. Then, flood from the robot pose
    a region to be regarded as reachable by the robot.

    If new obstacles are detected that are not present in a given
    grid map, the flooded area will replace the same area in the given grid
    map. This updates the grid map with new point cloud observation.

    Args:
        pcd (Open3D point cloud object)
        robot_position (tuple): x, y, z position of the robot (or could be 2D x, y)
        existing_map (GridMap2): existing grid map we want to update
        kwargs: paramters of the algorithm, including
            'layout_cut', 'floor_cut', 'grid_size' etc.
    Returns:
        GridMap2
    """
    # The height above which the points indicate nicely the layout of the room
    # while preserving big obstacles like tables.
    layout_cut = kwargs.get("layout_cut", 0.65)

    # We will regard points with z within layout_cut +/- floor_cut
    # to be the points that represent the floor.
    floor_cut = kwargs.get("floor_cut", 0.15)

    # length (in meters) of a grid in the grid map.
    grid_size = kwargs.get("grid_size", 0.25)

    # flood brush size: When flooding, we use a brush. If the brush
    # doesn't fit (it hits an obstacle), then the flood won't continue there.
    # 'brush_size' is the length of the square brush in meters. Intuitively,
    # this defines the width of the minimum pathway the robot can go through.
    brush_size = kwargs.get("brush_size", 0.5)

    # Because the map represented by the point cloud could be very large,
    # or even border-less, we want to constrain the grid-map we are building
    # or updating to be of a certain size. This is the radius of the region
    # we will build/update, in meters.
    region_size = kwargs.get("region_size", 10.0)

    # grid map name
    name = kwargs.get("name", "grid_map2")

    # whether to debug (show a visualiation)
    debug = kwargs.get("debug", False)

    # First: remove points below layout cut
    points = np.asarray(pcd.points)
    low_points_filter = np.less(points[:, 2], layout_cut)  # points below layout cut: will discard
    points = points[np.logical_not(low_points_filter)]  # points at or above layout cut

    # Second: identify points for the floor
    xmin, ymin, zmin = np.min(points, axis=0)
    floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)

    # Third, map points to POMDP space. If 'existing_map' is given, use it to do this.
    # Otherwise, the origin will be the minimum of points in the point cloud. This should
    # result in 2D points with integer coordinates.
    grid_points = []
    if existing_map is not None:
        for p in points:
            gp = existing_map.to_grid_pos(p[0], p[1])
            grid_points.append(gp)
        # also computer robot position on the grid map for later use
        grid_robot_position = existing_map.to_grid_pos(robot_position[0], robot_position[1])
    else:
        origin = (xmin, ymin)
        for p in points:
            gp = convert(p, Frame.WORLD, Frame.POMDP_SPACE,
                         region_origin=origin,
                         search_space_resolution=grid_size)
            grid_points.append(gp)
        grid_robot_position = convert(robot_position[:2], Frame.WORLD, Frame.POMDP_SPACE,
                                      region_origin=origin,
                                      search_space_resolution=grid_size)

    grid_points = np.asarray(grid_points)
    grid_points[:, 2] = 0  # suppress z

    # Fourth: build the reachable positions on the floor.
    # Start with the floors filter, which should still apply.
    grid_floor_points = grid_points[floor_points_filter]
    # Now flood from the robot position, with radius
    grid_floor_points = flood_fill_2d(grid_floor_points, (*grid_robot_position, 0),
                                      grid_brush_size=int(round(brush_size/grid_size)),
                                      flood_radius=int(round(region_size/grid_size/2)))

    # Fifth: build the obstacles and free locations: grid points are just obstacles
    # grid points on the floor that are not obstacles are free locations
    obstacles = set((gp[0], gp[1]) for gp in grid_points)
    free_locations = set((gp[0], gp[1]) for gp in grid_floor_points
                         if (gp[0], gp[1]) not in obstacles)

    ## Debugging
    if debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(grid_floor_points))
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(grid_floor_points), 3), (0.8, 0.8, 0.8)))

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.asarray(grid_points))
        pcd2.colors = o3d.utility.Vector3dVector(np.full((len(grid_points), 3), (0.2, 0.2, 0.2)))
        pcd2.points.append(np.asarray([*grid_robot_position, 0]))
        pcd2.colors.append([0.0, 0.8, 0.0])
        o3d.visualization.draw_geometries([pcd, pcd2])

    # Sixth: update existing map, or build new map
    if existing_map is not None:
        existing_map.update_region(obstacles, free_locations)
        return existing_map
    else:
        grid_map = GridMap2(name=name, obstacles=obstacles, free_locations=free_locations,
                            world_origin=origin, grid_size=grid_size, labels=None)
        return grid_map
