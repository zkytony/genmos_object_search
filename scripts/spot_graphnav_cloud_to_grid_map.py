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
import cv2
import open3d as o3d
from tqdm import tqdm
from collections import deque

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import sloop_ros.msg
import message_filters
from rbd_spot_perception.msg import GraphNavWaypointArray

from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.ros.grid_map_utils import grid_map_to_ros_msg
from sloop_object_search.utils.math import remap, in_region
from sloop_object_search.utils.visual import GridMapVisualizer

GRID_MAP_PUBLISHED = False

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

def flood_fill(grid_points, seed_point, brush_size=2):
    """
    Given a numpy array of points that are supposed to be on a grid map,
    and a "seed point", flood fill by adding more grid points that are
    in empty locations to the set of grid points, starting from the seed
    point. Will not modify 'grid_points' but returns a new array.

    brush_size (int): The length of a square brush which will
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
        brush_points = _neighbors(point, d=max(1, brush_size//2))
        new_points.update(brush_points)
        if not any(bp in grid_points_set for bp in brush_points):
            # This brush stroke fits; we will consider all brush points
            # as potential free cells - i.e. neighbors
            for neighbor_point in brush_points:
                if neighbor_point not in visited:
                    if in_region(neighbor_point, _ranges):
                        worklist.append(neighbor_point)
                        visited.add(neighbor_point)
    return np.array(list(map(np.array, grid_points_set | new_points)))


def pcd_to_grid_map(pcd, waypoints, **kwargs):
    """
    Given an Open3D point cloud object, output a GridMap object
    as the 2D projection of the point cloud.

    pcd (Open3D point cloud object)
    waypoints (numpy.array): L x 3 array where L is the number of waypoints;
        each row is a waypoint's position.
    """
    # The height above which the points indicate nicely the layout of the room
    # while preserving big obstacles like tables.
    layout_cut = kwargs.get("layout_cut", 0.65)

    # We will regard points with z within layout_cut +/- floor_cut
    # to be the points that represent the floor.
    floor_cut = kwargs.get("floor_cut", 0.15)

    # length (in meters) of a grid in the grid map.
    grid_size = kwargs.get("grid_size", 0.25)

    # percentage of waypoints to be sampled and used as seeds for flooding
    pct_waypoint_seeds = kwargs.get("pct_waypoint_seeds", 0.25)

    # grid map name
    name = kwargs.get("name", "grid_map")

    # whether to debug (show a visualiation)
    debug = kwargs.get("debug", False)

    # First, filter points by cutting those points below the layout.
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

        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(np.asarray(waypoints_grid_coords))
        waypoint_colors = np.full((waypoints_grid_coords.shape[0], 3), (0.0, 0.8, 0.0))
        waypoint_colors[selected_waypoints_indices] = np.array([0.8, 0.0, 0.0])
        pcd3.colors = o3d.utility.Vector3dVector(np.asarray(waypoint_colors))
        o3d.visualization.draw_geometries([pcd, pcd2, pcd3])

    grid_map = GridMap(width, length,
                       grid_map_obstacle_positions,
                       free_locations=grid_map_reachable_positions,
                       name=name,
                       ranges_in_metric=(metric_gx_range, metric_gy_range),
                       grid_size=grid_size)

    if debug:
        # Do a test: plot waypoints on the grid map
        waypoints_gx = remap(waypoints_grid_coords[:, 0], metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
        waypoints_gy = remap(waypoints_grid_coords[:, 1], metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)
        wyps = set(zip(waypoints_gx, waypoints_gy))

        viz = GridMapVisualizer(grid_map=grid_map, res=10)
        img = viz.render()
        img = viz.highlight(img, wyps, color=(120, 30, 30))
        img = viz.highlight(img, [(0, 2)], color=(80, 100, 230))
        viz.show_img(img, flip_horizontally=True)

    return grid_map

def waypoints_msg_to_arr(waypoints_msg):
    """converts a GraphNavWaypointArray message into a numpy array"""
    arr = np.array([[wp_msg.pose_sf.position.x,
                     wp_msg.pose_sf.position.y,
                     wp_msg.pose_sf.position.z]
                    for wp_msg in waypoints_msg.waypoints])
    return arr

class GraphNavPointCloudToGridMapPublisher:
    def __init__(self, args):
        rospy.init_node("graphnav_cloud_to_grid_map")
        self._debug = args.debug
        self.latch = not args.updating
        self.map_name = args.name
        self.grid_map_pub = rospy.Publisher(args.grid_map_topic, sloop_ros.msg.GridMap2d,
                                            queue_size=10, latch=self.latch)

        self.pcl_sub = message_filters.Subscriber(args.point_cloud_topic, PointCloud2)
        self.wyp_sub = message_filters.Subscriber(args.waypoint_topic, GraphNavWaypointArray)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pcl_sub, self.wyp_sub], 10, 0.2)  # allow 0.2s difference
        self.ts.registerCallback(self._cloud_waypoints_callback)

    def _cloud_waypoints_callback(self, cloud_msg, waypoints_msg):
        # Convert PointCloud2 message to Open3D point cloud
        # we will obtain a numpy array with each row being a waypoint's position
        print("Received point cloud and waypoints messages")
        waypoints_array = waypoints_msg_to_arr(waypoints_msg)
        points_raw_array = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
        points_array = ros_numpy.point_cloud2.get_xyz_points(points_raw_array)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array)

        # convert pcd to grid map
        grid_map = pcd_to_grid_map(pcd, waypoints_array, name=self.map_name, debug=self._debug)
        print("Grid Map created!")

        # Publish grid map as message
        grid_map_msg = grid_map_to_ros_msg(grid_map)
        self.grid_map_pub.publish(grid_map_msg)
        print("Published Grid Map")
        if self.latch:
            # we are done. close the subscribers
            self.ts.callbacks = {}  # unregister callback
            self.pcl_sub.sub.unregister()
            self.wyp_sub.sub.unregister()


def main():
    parser = argparse.ArgumentParser("PointCloud2 to GridMap2d")
    parser.add_argument("--point-cloud-topic", type=str, help="name of point cloud topic to subscribe to",
                        default="/graphnav_map_publisher/graphnav_points")
    parser.add_argument("--waypoint-topic", type=str, help="name of the topic for GraphNav waypoints",
                        default="/graphnav_waypoints")
    parser.add_argument("--grid-map-topic", type=str, help="name of grid map topic to publish at",
                        default="/graphnav_gridmap")
    parser.add_argument("--updating", action="store_true",
                        help="Keeps subscribing to point cloud and update the grid map; Otherwise, publishes once and latches.")
    parser.add_argument("--name", type=str, help="name of the grid map",
                        required=True)
    parser.add_argument("--debug", action="store_true", help="Debug grid map generation")
    args, _ = parser.parse_known_args()
    gmpub = GraphNavPointCloudToGridMapPublisher(args)
    rospy.spin()

if __name__ == "__main__":
    main()
