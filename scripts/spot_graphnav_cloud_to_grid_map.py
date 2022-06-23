#!/usr/bin/env python
#
# Subscribes to a PointCloud2 topic and a GraphNavWayPointArray topic
# and converts it to sloop_ros/GridMap2d, and publishes it.
#
# Note that this is script specific for Spot.
import time
import argparse
import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import deque

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import sloop_ros.msg
import message_filters
from rbd_spot_perception.msg import GraphNavWaypointArray

from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.utils.math import remap, in_region
from sloop_object_search.utils.visual import GridMapVisualizer


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

def flood_fill(grid_points, seed_point, brush_size=1):
    """
    Given a numpy array of points that are supposed to be on a grid map,
    and a "seed point", flood fill by adding more grid points that are
    in empty locations to the set of grid points, starting from the seed
    point. Will not modify 'grid_points' but returns a new array.

    brush_size (int): The half of the length of a square brush which will
        be used to fill out the empty spaces.
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
    visited = set()
    new_points = set()
    while len(worklist) > 0:
        point = worklist.popleft()
        # Imagine placing a square brush centered at the point.
        # We assume that the point always represents a valid free cell
        brush_points = _neighbors(point, d=brush_size)
        new_points.update(brush_points)
        if not any(bp in grid_points_set for bp in brush_points):
            # This brush stroke fits; we will consider all brush points
            # as potential free cells - i.e. neighbors
            for neighbor_point in brush_points:
                if neighbor_point not in visited:
                    if in_region(neighbor_point, _ranges):
                        worklist.append(neighbor_point)
                        visited.add(neighbor_point)
    return np.array(list(map(np.array, new_points)))


def pcd_to_grid_map(pcd, waypoints, **kwargs):
    """
    Given an Open3D point cloud object, output a GridMap object
    as the 2D projection of the point cloud.

    pcd (Open3D point cloud object)
    waypoints (numpy.array): L x 3 array where L is the number of waypoints;
        each row is a waypoint's position.
    """
    # We will regard points with z within layout_cut +/- floor_cut
    # to be the points that represent the floor.
    floor_cut = kwargs.get("floor_cut", 0.15)

    # The height above which the points indicate nicely the layout of the room
    # while preserving big obstacles like tables.
    layout_cut = kwargs.get("layout_cut", 0.5)

    # length (in meters) of a grid in the grid map.
    grid_size = kwargs.get("grid_size", 0.25)

    # # number of waypoint samples as seeds for flooding
    num_waypoint_seeds = kwargs.get("num_waypoint_seeds", 1)

    points = np.asarray(pcd.points)
    bad_points_filter = np.less(points[:, 2], layout_cut)
    points = points[np.logical_not(bad_points_filter)]
    xmin, ymin, zmin = np.min(points, axis=0)
    floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)
    floor_points = points[floor_points_filter]

    # We will first convert floor_points into an integer-coordinated grid map.
    floor_grid_coords = proj_to_grid_coords(floor_points, grid_size=grid_size)

    # also, convert waypoints into an integer-coordinated grid map
    waypoints_grid_coords = proj_to_grid_coords(waypoints, grid_size=grid_size)

    # Now, flood fill the floor_grid_coords with waypoints; we will select
    # way points that are of some distance away from each other.
    selected_waypoints_indices = [15]#np.random.choice(len(waypoints_grid_coords), num_waypoint_seeds)
    selected_waypoints = waypoints_grid_coords[15].reshape(1, -1)#[selected_waypoints_indices]
    for wp in tqdm(selected_waypoints): #selected_waypoints:
        floor_grid_coords = flood_fill(floor_grid_coords, wp)

    ## Debugging
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(floor_grid_coords))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(waypoints_grid_coords))
    waypoint_colors = np.full((waypoints_grid_coords.shape[0], 3), (0.0, 0.8, 0.0))
    waypoint_colors[selected_waypoints_indices] = np.array([0.8, 0.0, 0.0])
    pcd2.colors = o3d.utility.Vector3dVector(np.asarray(waypoint_colors))
    o3d.visualization.draw_geometries([pcd, pcd2])
    exit()


# def _make_grid_map(pcd, ceiling_cut=1.0,
#                     floor_cut=0.1,
#                     grid_size=0.25,
#                     debug=True):
#     """
#     Args:
#         pcd (Open3D PointCloud)
#         ceiling_cut (float): The points within `ceiling_cut` range (in meters) from the ymax
#             will be regarded as "ceiling"; You may want to set this number so
#             that the hanging lights are excluded.
#         floor_cut: same as ceiling_cut, but for floors. Set this to 0.4 for FloorPlan201
#     Returns:
#         GridMap

#     Note:
#         this code is based on thortils.map3d
#     """
#     downpcd = pcd.voxel_down_sample(voxel_size=0.25)
#     points = np.asarray(downpcd.points)



#     xmax, ymax, zmax = np.max(points, axis=0)
#     xmin, ymin, zmin = np.min(points, axis=0)

#     # Boundary points;
#     # Note: aggressively cutting ceiling and floor points;
#     # This may not be desirable if you only want to exclude
#     # points corresponding to the lights (this might be achievable
#     # by a combination of semantic segmantation and projection;
#     # left as a todo).
#     floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)
#     ceiling_points_filter = np.isclose(points[:,2], zmax, atol=ceiling_cut)
#     xwalls_min_filter = np.isclose(points[:,0], xmin, atol=0.05)
#     xwalls_max_filter = np.isclose(points[:,0], xmax, atol=0.05)
#     ywalls_min_filter = np.isclose(points[:,1], ymin, atol=0.05)
#     ywalls_max_filter = np.isclose(points[:,1], ymax, atol=0.05)
#     boundary_filter = np.any([floor_points_filter,
#                               ceiling_points_filter,
#                               xwalls_min_filter,
#                               xwalls_max_filter,
#                               ywalls_min_filter,
#                               ywalls_max_filter], axis=0)
#     not_boundary_filter = np.logical_not(boundary_filter)

#     # The simplest 2D grid map is Floor + Non-boundary points in 2D
#     # The floor points will be reachable, and the other ones are not.
#     # Points that will map to grid locations, but origin is NOT at (0,0);
#     # A lot of this code is borrowed from thortils.scene.convert_scene_to_grid_map.
#     map_points_filter = np.any([floor_points_filter,
#                                 not_boundary_filter], axis=0)
#     # The coordinates in points may be negative;
#     metric_grid_points = (points[map_points_filter] / grid_size).astype(int)
#     metric_gx = metric_grid_points[:,0]
#     metric_gy = metric_grid_points[:,1]
#     width = max(metric_gx) - min(metric_gx) + 1
#     length = max(metric_gy) - min(metric_gy) + 1
#     metric_gy = -metric_gy
#     metric_gx_range = (min(metric_gx), max(metric_gx) + 1)
#     metric_gy_range = (min(metric_gy), max(metric_gy) + 1)
#     # remap coordinates to be nonnegative (origin AT (0,0))
#     gx = remap(metric_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
#     gy = remap(metric_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)

#     gx_range = (min(gx), max(gx)+1)
#     gy_range = (min(gy), max(gy)+1)

#     # Little test: can convert back
#     try:
#         assert all(remap(gx, gx_range[0], gx_range[1], metric_gx_range[0], metric_gx_range[1]).astype(int) == metric_gx)
#         assert all(remap(gy, gy_range[0], gy_range[1], metric_gy_range[0], metric_gy_range[1]).astype(int) == metric_gy)
#     except AssertionError as ex:
#         print("Unable to remap coordinates")
#         raise ex

#     metric_reachable_points = points[floor_points_filter]
#     metric_reachable_points[:,2] = 0
#     metric_reachable_grid_points = (metric_reachable_points / grid_size).astype(int)
#     metric_reachable_gx = metric_reachable_grid_points[:,0]
#     metric_reachable_gy = metric_reachable_grid_points[:,1]
#     metric_reachable_gy = -metric_reachable_gy  # see [**] #length

#     metric_obstacle_points = points[not_boundary_filter]
#     metric_obstacle_points[:,2] = 0
#     metric_obstacle_grid_points = (metric_obstacle_points / grid_size).astype(int)
#     metric_obstacle_gx = metric_obstacle_grid_points[:,0]
#     metric_obstacle_gy = metric_obstacle_grid_points[:,1]
#     metric_obstacle_gy = -metric_obstacle_gy  # see [**] length

#     # For Debugging
#     if debug:
#         reachable_colors = np.full((metric_reachable_points.shape[0], 3), (0.6, 0.6, 0.6))
#         obstacle_colors = np.full((metric_obstacle_points.shape[0], 3), (0.2, 0.2, 0.2))

#         # We now grab points
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(np.asarray(metric_reachable_points))
#         pcd.colors = o3d.utility.Vector3dVector(np.asarray(reachable_colors))

#         # We now grab points
#         pcd2 = o3d.geometry.PointCloud()
#         pcd2.points = o3d.utility.Vector3dVector(np.asarray(metric_obstacle_points))
#         pcd2.colors = o3d.utility.Vector3dVector(np.asarray(obstacle_colors))
#         o3d.visualization.draw_geometries([pcd, pcd2])

#     gx_reachable = remap(metric_reachable_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
#     gy_reachable = remap(metric_reachable_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)
#     gx_obstacles = remap(metric_obstacle_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
#     gy_obstacles = remap(metric_obstacle_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)

#     all_positions = set((x,y) for x in range(width) for y in range(length))
#     grid_map_reachable_positions = set(zip(gx_reachable, gy_reachable))
#     grid_map_obstacle_positions = set(zip(gx_obstacles, gy_obstacles))

#     grid_map = GridMap(width, length,
#                        grid_map_obstacle_positions,
#                        unknown=(all_positions\
#                                 - grid_map_obstacle_positions\
#                                 - grid_map_reachable_positions),
#                        ranges_in_metric=(metric_gx_range, metric_gy_range),
#                        grid_size=grid_size)
#     return grid_map

def waypoints_msg_to_arr(waypoints_msg):
    arr = np.array([[wp_msg.pose_sf.position.x,
                     wp_msg.pose_sf.position.y,
                     wp_msg.pose_sf.position.z]
                    for wp_msg in waypoints_msg.waypoints])
    return arr

def _cloud_waypoints_callback(cloud_msg, waypoints_msg, args):
    # Convert PointCloud2 message to Open3D point cloud
    # we will obtain a numpy array with each row being a waypoint's position
    print("Received point cloud and waypoints messages")
    waypoints_array = waypoints_msg_to_arr(waypoints_msg)
    points_raw_array = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
    points_array = ros_numpy.point_cloud2.get_xyz_points(points_raw_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)

    # convert pcd to grid map
    grid_map = pcd_to_grid_map(pcd, waypoints_array)
    print("Grid Map created!")
    grid_map_pub = args[0]

    viz = GridMapVisualizer(grid_map=grid_map,
                            res=5)
    viz.show_img(viz.render())
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser("PointCloud2 to GridMap2d")
    parser.add_argument("--point-cloud-topic", type=str, help="name of point cloud topic to subscribe to",
                        default="/graphnav_map_publisher/graphnav_points")
    parser.add_argument("--waypoint-topic", type=str, help="name of the topic for GraphNav waypoints",
                        default="/graphnav_waypoints")
    parser.add_argument("--grid-map-topic", type=str, help="name of grid map topic to publish at",
                        default="/graphnav_gridmap")
    args = parser.parse_args()

    rospy.init_node("graphnav_cloud_to_grid_map")
    grid_map_pub = rospy.Publisher(args.grid_map_topic, sloop_ros.msg.GridMap2d, queue_size=10)

    pcl_sub = message_filters.Subscriber(args.point_cloud_topic, PointCloud2)
    wyp_sub = message_filters.Subscriber(args.waypoint_topic, GraphNavWaypointArray)
    ts = message_filters.ApproximateTimeSynchronizer([pcl_sub, wyp_sub], 10, 0.2)  # allow 0.2s difference
    ts.registerCallback(_cloud_waypoints_callback, (grid_map_pub,))


    rospy.spin()

if __name__ == "__main__":
    main()
