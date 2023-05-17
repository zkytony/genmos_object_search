#!/usr/bin/env python
# This is a simple script that publishes a static point cloud for the map used
# for the simple sim test.
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from ament_index_python.packages import get_package_share_directory
import ros2_numpy

MAP_POINTS_FILE = os.path.join(get_package_share_directory("genmos_object_search_ros2"),
                               "tests", "simple_sim", "lab121_lidar_points.npy")

class PointCloudPublisher(Node):

    def __init__(self):
        # Initialize the node with the name 'point_cloud_publisher'
        super().__init__('point_cloud_publisher')

        # Create a publisher on the 'point_cloud' topic, using the 'PointCloud2' message type
        self.publisher_ = self.create_publisher(PointCloud2, 'graphnav_points', 10)

        # Load the point cloud data from the .npy file
        self.get_logger().info(f"Loading numpy data from: {MAP_POINTS_FILE}")
        point_cloud_data = np.load(MAP_POINTS_FILE)
        point_cloud_xyz = np.zeros(len(point_cloud_data), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])
        point_cloud_xyz['x'] = point_cloud_data[:, 0]
        point_cloud_xyz['y'] = point_cloud_data[:, 1]
        point_cloud_xyz['z'] = point_cloud_data[:, 2]
        self.point_cloud_xyz = point_cloud_xyz

        self.declare_parameter('frame_id', 'graphnav_map')
        self.frame_id = self.get_parameter("frame_id").value

        # Convert the numpy array to a PointCloud2 message
        self.point_cloud_msg = ros2_numpy.point_cloud2.array_to_pointcloud2(
            self.point_cloud_xyz, stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

        # Set up a timer to call the 'timer_callback' method at a rate of 4 Hz
        self.timer_period = 0.25  # seconds (4 Hz)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        self.point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        # Publish the PointCloud2 message
        self.publisher_.publish(self.point_cloud_msg)
        self.get_logger().info('Publishing point cloud')


def main(args=None):
    rclpy.init(args=args)

    # Create a PointCloudPublisher node
    node = PointCloudPublisher()

    while rclpy.ok():
        rclpy.spin_once(node)
    # Shutdown and cleanup the node
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
