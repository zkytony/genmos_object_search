# To run this test, run
#
# ros2 topic pub "test1" std_msgs/msg/String "data: 'hello'
# ros2 topic pub "test2" std_msgs/msg/String "data: 'hello'
import rclpy
from rclpy.node import Node
import std_msgs.msg

from genmos_ros2 import ros2_utils


def test():
    rclpy.init()
    ros2_utils.wait_for_messages(
        ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2)
    ros2_utils.wait_for_messages(
        ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2)
    rclpy.shutdown()

if __name__ == "__main__":
    test()
