#!/usr/bin/env python
#
# Subscribes to a PointCloud2 topic and converts it to
# sloop_ros/GridMap2d, and publishes it.
import argparse
import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import sloop_ros.msg
from sloop_object_search.oopomdp.models.grid_map import GridMap
from sloop_object_search.utils.math import remap


def pcd_to_grid_map(pcd,
                    ceiling_cut=1.0,
                    floor_cut=0.1,
                    grid_size=0.25):
    """
    Args:
        pcd (Open3D PointCloud)
        ceiling_cut (float): The points within `ceiling_cut` range (in meters) from the ymax
            will be regarded as "ceiling"; You may want to set this number so
            that the hanging lights are excluded.
        floor_cut: same as ceiling_cut, but for floors. Set this to 0.4 for FloorPlan201
    Returns:
        GridMap

    Note:
        this code is based on thortils.map3d
    """
    debug = True
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(downpcd.points)

    xmax, ymax, zmax = np.max(points, axis=0)
    xmin, ymin, zmin = np.min(points, axis=0)

    # Boundary points;
    # Note: aggressively cutting ceiling and floor points;
    # This may not be desirable if you only want to exclude
    # points corresponding to the lights (this might be achievable
    # by a combination of semantic segmantation and projection;
    # left as a todo).
    floor_points_filter = np.isclose(points[:,1], ymin, atol=floor_cut)
    ceiling_points_filter = np.isclose(points[:,1], ymax, atol=ceiling_cut)
    xwalls_min_filter = np.isclose(points[:,0], xmin, atol=0.05)
    xwalls_max_filter = np.isclose(points[:,0], xmax, atol=0.05)
    zwalls_min_filter = np.isclose(points[:,2], zmin, atol=0.05)
    zwalls_max_filter = np.isclose(points[:,2], zmax, atol=0.05)
    boundary_filter = np.any([floor_points_filter,
                              ceiling_points_filter,
                              xwalls_min_filter,
                              xwalls_max_filter,
                              zwalls_min_filter,
                              zwalls_max_filter], axis=0)
    not_boundary_filter = np.logical_not(boundary_filter)

    # The simplest 2D grid map is Floor + Non-boundary points in 2D
    # The floor points will be reachable, and the other ones are not.
    # Points that will map to grid locations, but origin is NOT at (0,0);
    # A lot of this code is borrowed from thortils.scene.convert_scene_to_grid_map.
    map_points_filter = np.any([floor_points_filter,
                                not_boundary_filter], axis=0)
    # The coordinates in points may be negative;
    metric_grid_points = (points[map_points_filter] / grid_size).astype(int)
    metric_gx = metric_grid_points[:,0]
    metric_gy = metric_grid_points[:,2]
    width = max(metric_gx) - min(metric_gx) + 1
    length = max(metric_gy) - min(metric_gy) + 1
    metric_gx_range = (min(metric_gx), max(metric_gx) + 1)
    metric_gy_range = (min(metric_gy), max(metric_gy) + 1)
    # remap coordinates to be nonnegative (origin AT (0,0))
    gx = remap(metric_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
    gy = remap(metric_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)

    gx_range = (min(gx), max(gx)+1)
    gy_range = (min(gy), max(gy)+1)

    # Little test: can convert back
    try:
        assert all(remap(gx, gx_range[0], gx_range[1], metric_gx_range[0], metric_gx_range[1]).astype(int) == metric_gx)
        assert all(remap(gy, gy_range[0], gy_range[1], metric_gy_range[0], metric_gy_range[1]).astype(int) == metric_gy)
    except AssertionError as ex:
        print("Unable to remap coordinates")
        raise ex

    metric_reachable_points = points[floor_points_filter]
    metric_reachable_points[:,1] = 0
    metric_reachable_grid_points = (metric_reachable_points / grid_size).astype(int)
    metric_reachable_gx = metric_reachable_grid_points[:,0]
    metric_reachable_gy = metric_reachable_grid_points[:,2]
    metric_reachable_gy = -metric_reachable_gy  # see [**] #length

    metric_obstacle_points = points[not_boundary_filter]
    metric_obstacle_points[:,1] = 0
    metric_obstacle_grid_points = (metric_obstacle_points / grid_size).astype(int)
    metric_obstacle_gx = metric_obstacle_grid_points[:,0]
    metric_obstacle_gy = metric_obstacle_grid_points[:,2]
    metric_obstacle_gy = -metric_obstacle_gy  # see [**] length

    # For Debugging
    if debug:
        reachable_colors = np.full((metric_reachable_points.shape[0], 3), (0.7, 0.7, 0.7))
        obstacle_colors = np.full((metric_obstacle_points.shape[0], 3), (0.2, 0.2, 0.2))

        # We now grab points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(metric_reachable_points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(reachable_colors))

        # We now grab points
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.asarray(metric_obstacle_points))
        pcd2.colors = o3d.utility.Vector3dVector(np.asarray(obstacle_colors))
        o3d.visualization.draw_geometries([pcd, pcd2])

    gx_reachable = remap(metric_reachable_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
    gy_reachable = remap(metric_reachable_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)
    gx_obstacles = remap(metric_obstacle_gx, metric_gx_range[0], metric_gx_range[1], 0, width).astype(int)
    gy_obstacles = remap(metric_obstacle_gy, metric_gy_range[0], metric_gy_range[1], 0, length).astype(int)

    all_positions = set((x,y) for x in range(width) for y in range(length))
    grid_map_reachable_positions = set(zip(gx_reachable, gy_reachable))
    grid_map_obstacle_positions = set(zip(gx_obstacles, gy_obstacles))

    grid_map = GridMap(width, length,
                       grid_map_obstacle_positions,
                       unknown=(all_positions\
                                - grid_map_obstacle_positions\
                                - grid_map_reachable_positions),
                       name=scene,
                       ranges_in_thor=(metric_gx_range, metric_gy_range),
                       grid_size=grid_size)
    return grid_map




def _point_cloud_callback(msg, args):
    # Convert PointCloud2 message to Open3D point cloud
    import pdb; pdb.set_trace()
    points_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    points = ros_numpy.point_cloud2.get_xyz_points(points_array)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # convert pcd to grid map

    grid_map = pcd_to_grid_map(pcd)

    print("HELLO!")
    grid_map_pub = args[0]



def main():
    parser = argparse.ArgumentParser("PointCloud2 to GridMap2d")
    parser.add_argument("point_cloud_topic", type=str, help="name of point cloud topic to subscribe to")
    parser.add_argument("grid_map_topic", type=str, help="name of grid map topic to publish at")
    args = parser.parse_args()

    rospy.init_node("point_cloud_to_grid_map")
    grid_map_pub = rospy.Publisher(args.grid_map_topic, sloop_ros.msg.GridMap2d, queue_size=10)
    rospy.Subscriber(args.point_cloud_topic, PointCloud2, _point_cloud_callback, (grid_map_pub,))
    rospy.spin()

if __name__ == "__main__":
    main()
