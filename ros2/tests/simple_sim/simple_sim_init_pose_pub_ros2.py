#!/usr/bin/env python
import rclpy
import threading
from rclpy.time import Duration
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from genmos_ros2 import ros2_utils
from genmos_object_search.utils.math import euler_to_quat

class SimpleSimEnvInitPosePublisher(Node):
    def __init__(self):
        super().__init__("simple_sim_init_robot_pose_publisher")
        params = [("x", 0.0),
                  ("y", 0.0),
                  ("z", 0.0),
                  ("thx", 0.0),
                  ("thy", 0.0),
                  ("thz", 0.0),
                  ("world_frame", "graphnav_map"),
                  ("latch", True),
                  ("lifetime", 0)]
        ros2_utils.declare_params(self, params)
        ros2_utils.print_parameters(self, params)

        self.latch = self.get_parameter("latch").value
        if self.latch:
            qos_profile = ros2_utils.latch(depth=10)
            self._msg_is_published = False
        else:
            qos_profile = 10
        self.publisher = self.create_publisher(
            PoseStamped, "/simple_sim_env/init_robot_pose", qos_profile)
        x = self.get_parameter("x").value
        y = self.get_parameter("y").value
        z = self.get_parameter("z").value
        thx = self.get_parameter("thx").value
        thy = self.get_parameter("thy").value
        thz = self.get_parameter("thz").value
        qx, qy, qz, qw = euler_to_quat(thx, thy, thz)
        self._init_pose = (x,y,z,qx,qy,qz,qw)
        self.world_frame = self.get_parameter("world_frame").value
        self.lifetime = Duration(seconds=0)
        if self.get_parameter("lifetime").value != 0:
            self.lifetime = Duration(nanoseconds=self.get_parameter("lifetime").value * 1e9)
        self.timer = self.create_timer(0.5, self.publish_msg)


    def publish_msg(self):
        if self.latch and self._msg_is_published:
            return
        msg = ros2_utils.pose_tuple_to_pose_stamped(
            self._init_pose, self.world_frame, node=self)
        self.publisher.publish(msg)
        self.get_logger().info("published initial pose")
        self._msg_is_published = True


def main():
    rclpy.init()
    node = SimpleSimEnvInitPosePublisher()
    start_time = node.get_clock().now()
    times_up = False
    while not times_up and rclpy.ok():
        rclpy.spin_once(node)
        time_elapsed = node.get_clock().now() - start_time
        node.get_logger().info(f"{time_elapsed}, {node.lifetime}")
        if node.lifetime != Duration(seconds=0) and time_elapsed > node.lifetime:
            node.get_logger().info("Times up. Shutdown.")
            break
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
