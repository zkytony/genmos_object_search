import math
import numpy as np
import open3d as o3d
from collections import deque
from tqdm import tqdm

from .proto_utils import pointcloudproto_to_array
from sloop_object_search.utils.math import (remap, euclidean_dist,
                                            eucdist_multi, in_square, in_square_multi)
from sloop_object_search.utils.visual import GridMapVisualizer
from sloop_object_search.utils.conversion import Frame, convert
from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.oopomdp.models.search_region import SearchRegion2D


########### 2D search region ##############
def search_region_2d_from_occupancy_grid(occupancy_grid, robot_position, existing_search_region=None, **kwargs):
    pass


def search_region_2d_from_point_cloud(point_cloud, robot_position, existing_search_region=None, **kwargs):
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

    # build grid map
    search_region = pcd_to_search_region_2d(
        pcd, robot_position,
        existing_search_region=existing_search_region,
        debug=True, **kwargs)
    if existing_search_region is not None:
        # search_region should be the same as existing
        assert existing_search_region.grid_map == search_region.grid_map
        return existing_search_region
    else:
        print("grid map created!")
        return search_region

def flood_fill_2d(grid_points, seed_point, grid_brush_size=2, flood_region_size=None):
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
        flood_region_size (float): the maximum size (number of grids) of
            the flooding region which is a square.
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
                    if flood_region_size is not None:
                        if not in_square(neighbor_point, seed_point, flood_region_size):
                            continue  # skip this point: too far.
                    worklist.append(neighbor_point)
                    visited.add(neighbor_point)
    return np.array(list(grid_points_set | new_points))


def pcd_to_search_region_2d(pcd, robot_position, existing_search_region=None, **kwargs):
    """
    Given an Open3D point cloud object, the robot current pose, and
    optionally an existing grid map, output a SearchRegion2D object
    which contains a grid map (GridMap2) as the 2D projection of the point cloud.

    The algorithm works by first treating points above a certain
    height threshold (layout_cut) as points that form obstacles that
    identify the layout of the map. Then, flood from the robot pose
    a region to be regarded as reachable by the robot.

    Only points within the region will be considered to build the grid map;
    In other words, the region we flood will be the region we will build
    the grid map.  Essentially, we are building a grid map for a portion
    of the given point cloud within a square region centered at the robot position.

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
        SearchRegion2
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
    # or updating to be of a certain size. This is the size of the square region
    # we will build/update, in meters.
    region_size = kwargs.get("region_size", 10.0)

    # grid map name
    name = kwargs.get("name", "grid_map2")

    # whether to debug (show a visualiation)
    debug = kwargs.get("debug", False)

    # Remove points below layout cut
    points = np.asarray(pcd.points)
    low_points_filter = np.less(points[:, 2], layout_cut)  # points below layout cut: will discard
    points = points[np.logical_not(low_points_filter)]  # points at or above layout cut

    # Filter out points beyond region_size
    region_points_filter = in_square_multi(points[:, :2], robot_position[:2], region_size)
    points = points[region_points_filter]

    # Identify points for the floor
    xmin, ymin, zmin = np.min(points, axis=0)
    floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)

    # Map points to POMDP space. If 'existing_map' is given, use it to do this.
    # Otherwise, the origin will be the minimum of points in the point cloud. This should
    # result in 2D points with integer coordinates.
    grid_points = []
    if existing_search_region is not None:
        for p in points:
            gp = existing_search_region.to_grid_pos(p[0], p[1])
            grid_points.append((*gp, 0))
        # also computer robot position on the grid map for later use
        grid_robot_position = existing_search_region.to_grid_pos(robot_position[0], robot_position[1])
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

    # Build the reachable positions on the floor.
    # Start with the floors filter, which should still apply.
    grid_floor_points = grid_points[floor_points_filter]
    # Now flood from the robot position, with radius
    grid_floor_points = flood_fill_2d(grid_floor_points, (*grid_robot_position, 0),
                                      grid_brush_size=int(round(brush_size/grid_size)),
                                      flood_region_size=int(round(region_size/grid_size)))

    # Build the obstacles and free locations: grid points are just obstacles
    # grid points on the floor that are not obstacles are free locations
    obstacles = set((gp[0], gp[1]) for gp in grid_points)
    free_locations = set((gp[0], gp[1]) for gp in grid_floor_points
                         if (gp[0], gp[1]) not in obstacles)

    # Update existing map, or build new map
    if existing_search_region is not None:
        existing_search_region.grid_map.update_region(obstacles, free_locations)
        return_search_region = existing_search_region
    else:
        grid_map = GridMap2(name=name, obstacles=obstacles,
                            free_locations=free_locations, labels=None)
        return_search_region = SearchRegion2D(grid_map,
                                              region_origin=origin,
                                              grid_size=grid_size)

    ## Debugging
    if debug:
        # import pdb; pdb.set_trace()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(list(points)))
        # pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), (0.8, 0.8, 0.8)))
        # o3d.visualization.draw_geometries([pcd])

        resulting_map = return_search_region.grid_map

        pcd = o3d.geometry.PointCloud()
        freeloc_points = np.asarray(list(resulting_map.free_locations))
        freeloc_points = np.append(freeloc_points, np.zeros((len(freeloc_points), 1)), axis=1)
        pcd.points = o3d.utility.Vector3dVector(freeloc_points)
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(resulting_map.free_locations), 3), (0.8, 0.8, 0.8)))

        pcd2 = o3d.geometry.PointCloud()
        obloc_points = np.asarray(list(resulting_map.obstacles))
        obloc_points = np.append(obloc_points, np.zeros((len(obloc_points), 1)), axis=1)
        pcd2.points = o3d.utility.Vector3dVector(obloc_points)
        pcd2.colors = o3d.utility.Vector3dVector(np.full((len(resulting_map.obstacles), 3), (0.2, 0.2, 0.2)))
        pcd2.points.append([*grid_robot_position, 1])
        pcd2.colors.append([0.0, 0.8, 0.0])
        o3d.visualization.draw_geometries([pcd, pcd2])

    return return_search_region


########### 3D search region ##############
def search_region_3d_from_point_cloud(point_cloud, robot_position, existing_search_region=None, **kwargs):
    pass
