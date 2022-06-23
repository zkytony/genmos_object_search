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

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import sloop_ros.msg
import message_filters
from rbd_spot_perception.msg import GraphNavWaypointArray

from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.utils.math import remap
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




def pcd_to_grid_map(pcd, **kwargs):
    """
    Given an Open3D point cloud object, output a GridMap object
    as the 2D projection of the point cloud.

    pcd (Open3D point cloud object)
    """
    # We will regard points with z within layout_cut +/- floor_cut
    # to be the points that represent the floor.
    floor_cut = kwargs.get("floor_cut", 0.15)

    # The height above which the points indicate nicely the layout of the room
    # while preserving big obstacles like tables.
    layout_cut = kwargs.get("layout_cut", 0.5)

    # length (in meters) of a grid in the grid map.
    grid_size = kwargs.get("grid_size", 0.25)

    points = np.asarray(pcd.points)
    bad_points_filter = np.less(points[:, 2], layout_cut)
    points = points[np.logical_not(bad_points_filter)]
    xmin, ymin, zmin = np.min(points, axis=0)
    floor_points_filter = np.isclose(points[:,2], zmin, atol=floor_cut)
    floor_points = points[floor_points_filter]

    # We will first convert floor_points into an integer-coordinated grid map.
    floor_grid_coords = proj_to_grid_coords(floor_points, grid_size=grid_size)

    # floor_points[:,2] = 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(floor_grid_coords))
    o3d.visualization.draw_geometries([pcd])



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

def _cloud_waypoints_callback(cloud_msg, waypoints_msg, args):
    # Convert PointCloud2 message to Open3D point cloud
    print(waypoints_msg)
    points_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    points = ros_numpy.point_cloud2.get_xyz_points(points_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # convert pcd to grid map
    grid_map = pcd_to_grid_map(pcd)
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
