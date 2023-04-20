# This script illustrates the basic behavior of
# latching in ROS2
#
# Suppose:
#   pub -> /test (latches)
#            \---> sub
import rclpy
import std_msgs.msg
from genmos_ros2 import ros2_utils

def cb(msg):
    print(f"MESSAGE: {msg.data}")

def test():
    rclpy.init()
    node1 = rclpy.create_node("pub")
    pub = node1.create_publisher(std_msgs.msg.String, "test", ros2_utils.latch(depth=10))
    node2 = rclpy.create_node("sub")
    sub = node2.create_subscription(std_msgs.msg.String, "test", cb, ros2_utils.latch(depth=10))
    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(node1)
    executor.add_node(node2)
    pub.publish(std_msgs.msg.String(data="hello world"))
    executor.spin()
    node1.destroy_node()
    node2.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test()
