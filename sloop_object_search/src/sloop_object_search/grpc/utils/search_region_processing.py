import numpy as np
import open3d as o3d
from collections import deque
from tqdm import tqdm

from .proto_utils import pointcloudproto_to_array
from sloop_object_search.utils.math import remap, in_region, euclidean_dist
from sloop_object_search.utils.visual import GridMapVisualizer
from sloop_object_search.utils.conversion import Frame, convert
from sloop_object_search.oopomdp.models.grid_map import GridMap


def search_region_from_occupancy_grid():
    pass


def search_region_from_point_cloud(point_cloud, world_origin=None, is_3d=False, **kwargs):
    """
    The points in the given point cloud should correspond to static
    obstacles in the environment. The extent of this point cloud forms
    the extent of the search region.

    If the point_cloud is to be projected down to 2D, then we assume
    there is a "floor" (or flat plane) in the environment. "floor_cut"
    in kwargs specifies the height below which the points are considered
    to be part of the floor.

    'world_origin' is a point in the world frame that corresponds to (0,0) or
    (0,0,0) in the POMDP model of the world. If it is None, then the world_origin
    will be set to the point with minimum coordinates in the point cloud.

    Args:
        point_cloud (common_pb2.PointCloud): The input point cloud
        is_3d (bool): whether the search region will be 3D
    """
    if is_3d:
        pass
    else:
        search_region_2d_from_point_cloud(point_cloud, world_origin=None, **kwargs)

########### 2D search region ##############
def search_region_2d_from_point_cloud(point_cloud, world_origin=None, **kwargs):
    points_array = pointcloudproto_to_array(point_cloud)
    pcd = o3d.geometry.PointCloud()
    import pdb; pdb.set_trace()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    grid_map = pcd_to_grid_map(pcd, [], debug=True, **kwargs)
    print("grid map created!")

def proj_to_grid_coords(points, grid_size=0.25):
    """
    Given points (numpy array of shape Nx3), return a numpy
    array of shape Nx3 where each number is an integer, which
    represents the coordinate over a grid map where each grid
    has length 'grid_size'. The integer could be negative.
    The z axis of all resulting points will be set to 0.
    """
    metric_grid_points = (points / grid_size).astype(int)
    metric_grid_points[:,2] = 0
    return metric_grid_points

def flood_fill(grid_points, seed_point, grid_brush_size=2, flood_radius=None):
    """
    Given a numpy array of points that are supposed to be on a grid map,
    and a "seed point", flood fill by adding more grid points that are
    in empty locations to the set of grid points, starting from the seed
    point. Will not modify 'grid_points' but returns a new array.

    grid_brush_size (int): The length (number of grids) of a square brush
        which will be used to fill out the empty spaces.

    flood_radius (float): the maximum euclidean distance between a
        point in the flood and the seed point.
    """
    def _neighbors(p, d=1):
        # note that p is 3D, but we only care about x and y
        return set((p[0] + dx, p[1] + dy, p[2])
                   for dx in range(-d, d+1)
                   for dy in range(-d, d+1)
                   if not (dx == 0 and dy == 0))

    seed_point = tuple(seed_point)
    xmax, ymax, zmax = np.max(grid_points, axis=0)
    xmin, ymin, zmin = np.min(grid_points, axis=0)
    _ranges = ([xmin, xmax], [ymin, ymax], [zmin, zmax+1])
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
                    if in_region(neighbor_point, _ranges):
                        worklist.append(neighbor_point)
                        visited.add(neighbor_point)
    return np.array(list(map(np.array, grid_points_set | new_points)))


def pcd_to_grid_map(pcd, robot_position, existing_map=None, **kwargs):
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
        robot_position (tuple): x, y, z position of the robot
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
    # we will build/update, in meters
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

    # Fourth: build the reachable positions on the floor.
    # Start with the floors filter, which should still apply.
    grid_floor_points = np.asarray(grid_points)[floor_points_filter]
    # Now flood from the robot position, with radius
    grid_floor_points = flood_fill(grid_floor_points, grid_robot_position)


    ###############################################################################

    # First, filter points by cutting those points below the layout.
    points = np.asarray(pcd.points)
    bad_points_filter = np.less(points[:, 2], layout_cut)
    points = points[np.logical_not(bad_points_filter)]
    xmin, ymin, zmin = np.min(points, axis=0)
    floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)
    floor_points = points[floor_points_filter]

    # We will first convert floor_points into an integer-coordinated grid map.
    floor_grid_coords = proj_to_grid_coords(floor_points, grid_size=grid_size)

    # also, convert robot position into

    # also, convert waypoints into an integer-coordinated grid map
    if len(waypoints) > 0:
        waypoints_grid_coords = proj_to_grid_coords(waypoints, grid_size=grid_size)

        # Now, flood fill the floor_grid_coords with waypoints; we will select
        # way points that are of some distance away from each other.
        num_waypoint_seeds = int(len(waypoints_grid_coords) * pct_waypoint_seeds)
        np.random.seed(1010)
        selected_waypoints_indices = np.random.choice(len(waypoints_grid_coords), num_waypoint_seeds)
        selected_waypoints = waypoints_grid_coords[selected_waypoints_indices]
        for wp in tqdm(selected_waypoints):
            floor_grid_coords = flood_fill(floor_grid_coords, wp)

    # The floor points will be reachable points
    metric_reachable_grid_points = floor_grid_coords
    metric_reachable_gx = metric_reachable_grid_points[:,0]
    metric_reachable_gy = metric_reachable_grid_points[:,1]

    # For the obstacles, we get points above the floor - we already have them in points
    metric_obstacle_points = points
    metric_obstacle_points[:,2] = 0
    metric_obstacle_grid_points = (metric_obstacle_points / grid_size).astype(int)
    metric_obstacle_gx = metric_obstacle_grid_points[:,0]
    metric_obstacle_gy = metric_obstacle_grid_points[:,1]

    # obtain ranges
    metric_gx = metric_obstacle_grid_points[:,0]
    metric_gy = metric_obstacle_grid_points[:,1]
    width = max(metric_gx) - min(metric_gx) + 1
    length = max(metric_gy) - min(metric_gy) + 1
    metric_gx_range = (min(metric_gx), max(metric_gx) + 1)
    metric_gy_range = (min(metric_gy), max(metric_gy) + 1)

    # Now, we get points with 0-based coordinates, which are actual grid map points
    gx_reachable = remap(metric_reachable_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
    gy_reachable = remap(metric_reachable_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)
    gx_obstacles = remap(metric_obstacle_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
    gy_obstacles = remap(metric_obstacle_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)

    all_positions = set((x,y) for x in range(width) for y in range(length))
    grid_map_reachable_positions = set(zip(gx_reachable, gy_reachable))
    grid_map_obstacle_positions = set(zip(gx_obstacles, gy_obstacles))

    ## Debugging
    if debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(metric_reachable_grid_points))
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(metric_reachable_grid_points), 3), (0.8, 0.8, 0.8)))

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.asarray(metric_obstacle_grid_points))
        pcd2.colors = o3d.utility.Vector3dVector(np.full((len(metric_obstacle_grid_points), 3), (0.2, 0.2, 0.2)))

        if len(waypoints) > 0:
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(np.asarray(waypoints_grid_coords))
            waypoint_colors = np.full((waypoints_grid_coords.shape[0], 3), (0.0, 0.8, 0.0))
            waypoint_colors[selected_waypoints_indices] = np.array([0.8, 0.0, 0.0])
            pcd3.colors = o3d.utility.Vector3dVector(np.asarray(waypoint_colors))
            o3d.visualization.draw_geometries([pcd, pcd2, pcd3])
        else:
            o3d.visualization.draw_geometries([pcd, pcd2])

    grid_map = GridMap(width, length,
                       grid_map_obstacle_positions,
                       free_locations=grid_map_reachable_positions,
                       name=name,
                       ranges_in_metric=(metric_gx_range, metric_gy_range),
                       grid_size=grid_size)

    if debug:
        # Do a test: plot waypoints on the grid map
        if len(waypoints) > 0:
            waypoints_gx = remap(waypoints_grid_coords[:, 0], metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
            waypoints_gy = remap(waypoints_grid_coords[:, 1], metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)
            wyps = set(zip(waypoints_gx, waypoints_gy))

        viz = GridMapVisualizer(grid_map=grid_map, res=10)
        img = viz.render()
        img = viz.highlight(img, wyps, color=(120, 30, 30))
        img = viz.highlight(img, [(0, 2)], color=(80, 100, 230))
        viz.show_img(img, flip_horizontally=True)

    return grid_map
