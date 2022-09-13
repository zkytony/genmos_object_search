import math
import numpy as np
import open3d as o3d
from collections import deque
from tqdm import tqdm

from .proto_utils import pointcloudproto_to_array
from sloop_object_search.utils.math import (remap, euclidean_dist,
                                            eucdist_multi, in_square, in_square_multi)
from sloop_object_search.utils.conversion import Frame, convert
from sloop_object_search.utils import open3d_utils
from sloop_object_search.utils.algo import flood_fill_2d
from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.oopomdp.models.search_region import SearchRegion2D, SearchRegion3D
from sloop_object_search.oopomdp.models.octree_belief\
    import Octree, OctreeDistribution, OccupancyOctreeDistribution


########### 2D search region ##############
def search_region_2d_from_occupancy_grid(occupancy_grid, robot_position, existing_search_region=None, **kwargs):
    raise NotImplementedError()


def search_region_2d_from_point_cloud(point_cloud, robot_position, existing_search_region=None, **kwargs):
    """
    The points in the given point cloud should correspond to
    obstacles in the environment.

    The point_cloud is to be projected down to 2D; We assume
    there is a "floor" (or flat plane) in the environment. "floor_cut"
    in kwargs specifies the height below which the points are considered
    to be part of the floor.
    """
    points_array = pointcloudproto_to_array(point_cloud)

    # build grid map
    search_region = points_to_search_region_2d(
        points_array, robot_position,
        existing_search_region=existing_search_region,
        **kwargs)

    if existing_search_region is not None:
        # search_region should be the same as existing
        assert existing_search_region.grid_map == search_region.grid_map
        return_search_region = existing_search_region
    else:
        print("grid map created!")
        return_search_region = search_region

    # Labeling grid cells as reachable / reachable for topo / potential object locations
    # TODO: is this necessary?

    return return_search_region


########### 3D search region ##############
def search_region_3d_from_point_cloud(point_cloud, robot_position, existing_search_region=None, **kwargs):
    """
    The points in the given point cloud should correspond to
    obstacles in the environment.

    The point cloud will be converted into an octree with
    a given resolution. Note that having an octree of bigger
    than 64x64x64 may be slow. Optimization of octree implementation
    (e.g. in C++ with python binding) is pending.

    For now, if there is an existing search region, the octree
    of the search region will be discarded, but its origin is kept.
    """
    points_array = pointcloudproto_to_array(point_cloud)

    # size of the search region (in meters)
    default_sizes = np.max(points_array, axis=0) - np.min(points_array, axis=0)  # size of the search region in each axis
    region_size_x = kwargs.get("region_size_x", default_sizes[0])
    region_size_y = kwargs.get("region_size_y", default_sizes[1])
    region_size_z = kwargs.get("region_size_z", default_sizes[2])

    # robot position will be the center of the search region
    origin = np.array([robot_position[0] - region_size_x/2,
                       robot_position[1] - region_size_y/2,
                       robot_position[2] - region_size_z/2])

    sizes = np.array([region_size_x, region_size_y, region_size_z])

    # Size of one dimension of the space that the octree covers
    # Must be a power of two.
    octree_size = kwargs.get("octree_size", 64)

    # Meter of a side of a ground-level cube in the octree. This represents
    # the resolution of the octree when placed in the real world.
    search_space_resolution = kwargs.get("search_space_resolution", 0.15)

    # whether to debug (show a visualiation)
    debug = kwargs.get("debug", False)

    dimensions = (octree_size, octree_size, octree_size)

    origin_pomdp = convert(origin, Frame.WORLD, Frame.POMDP_SPACE,
                           region_origin=origin,
                           search_space_resolution=search_space_resolution)
    octree_dist = OccupancyOctreeDistribution(dimensions, (origin_pomdp, *(sizes / search_space_resolution)))

    # Either update existing search region, or create a brand new one.
    if existing_search_region is not None:
        # the octree of the search region will be discarded, but its origin is kept.
        search_region = existing_search_region
        search_region.update(octree_dist)
    else:
        search_region = SearchRegion3D(
            octree_dist, region_origin=origin,
            search_space_resolution=search_space_resolution)

    for p in points_array:
        g = search_region.to_octree_pos(p)
        if search_region.valid_voxel((*g, 1)):
            search_region.octree_dist[(*g, 1)] = 1  # set value to be 1
        else:
            if debug:
                print(f"Warning: voxel {g} is out of bound of the octree. Octree doesn't cover search region.")
    # debugging
    if debug:
        open3d_utils.draw_search_region3d(search_region, points=points_array)

    return search_region



########### Auxiliary functions for 2D Search Regeion ##############
def points_to_search_region_2d(points, robot_position, existing_search_region=None, **kwargs):
    """
    Given a Numpy array of points (N,3), the robot current pose, and
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

    points and robot position are in the world frame.

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
    # we will build/update, in meters. IF DON'T WANT THIS, SET IT TO NEGATIVE.
    region_size = kwargs.get("region_size", 10.0)

    # Search region init options
    init_options = dict(include_free=kwargs.get("include_free", True),
                        include_obstacles=kwargs.get("include_obstacles", False),
                        expansion_width=kwargs.get("expansion_width", 0.5))

    # grid map name
    name = kwargs.get("name", "grid_map2")

    # whether to debug (show a visualiation)
    debug = kwargs.get("debug", False)

    # Remove points below layout cut
    low_points_filter = np.less(points[:, 2], layout_cut)  # points below layout cut: will discard
    points = points[np.logical_not(low_points_filter)]  # points at or above layout cut

    # Filter out points beyond region_size
    if region_size > 0:
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
            gp = existing_search_region.to_grid_pos(p[:2])
            grid_points.append((*gp, 0))
        # also computer robot position on the grid map for later use
        grid_robot_position = existing_search_region.to_grid_pos(robot_position[:2])
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
    flood_region_size = None
    if region_size > 0:
        flood_region_size = int(round(region_size/grid_size))
    grid_floor_points = flood_fill_2d(grid_floor_points, (*grid_robot_position, 0),
                                      grid_brush_size=int(round(brush_size/grid_size)),
                                      flood_region_size=flood_region_size)

    # Build the obstacles and free locations: grid points are just obstacles
    # grid points on the floor that are not obstacles are free locations
    obstacles = set((gp[0], gp[1]) for gp in grid_points)
    free_locations = set((gp[0], gp[1]) for gp in grid_floor_points
                         if (gp[0], gp[1]) not in obstacles)

    # Update existing map, or build new map
    if existing_search_region is not None:
        existing_search_region.update(obstacles, free_locations)
        return_search_region = existing_search_region
    else:
        grid_map = GridMap2(name=name, obstacles=obstacles, free_locations=free_locations)
        return_search_region = SearchRegion2D(grid_map,
                                              region_origin=origin,
                                              grid_size=grid_size,
                                              **init_options)

    ## Debugging
    if debug:
        open3d_utils.draw_search_region2d(return_search_region,
                                          grid_robot_position=grid_robot_position,
                                          points=points)

    return return_search_region
