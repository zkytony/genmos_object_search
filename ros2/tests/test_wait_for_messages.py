# To run this test, run 'python test_wait_for_messages.py'
import threading

import rclpy
import std_msgs.msg
import geometry_msgs.msg
from rclpy.node import Node

from genmos_ros2 import ros2_utils


class NodeFoo(Node):
    def __init__(self):
        super().__init__("foo")
        self.pub = self.create_publisher(std_msgs.msg.String, "test1", ros2_utils.latch(depth=10))
        self.pub.publish(std_msgs.msg.String(data="hello from foo"))

class NodeBar(Node):
    def __init__(self):
        super().__init__("bar")
        self.pub = self.create_publisher(std_msgs.msg.String, "test2", ros2_utils.latch(depth=10))
        self.timer = self.create_timer(0.5, self.publish_msg)

    def publish_msg(self):
        self.pub.publish(std_msgs.msg.String(data="hello from bar"))


def test():
    rclpy.init()
    node_foo = NodeFoo()
    node_bar = NodeBar()
    node_wfm = rclpy.create_node("wait_for_messages")

    executor = rclpy.executors.MultiThreadedExecutor(4)
    executor.add_node(node_foo)
    executor.add_node(node_bar)
    executor.add_node(node_wfm)

    t_ex = threading.Thread(target=executor.spin, args=(), daemon=False)
    t_ex.start()

    print(ros2_utils.wait_for_messages(
        node_wfm, ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2, latched_topics={"/test1"}))
    print(ros2_utils.wait_for_messages(
        node_wfm, ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2, latched_topics={"/test1"}))
    # robot_pose_msg = ros2_utils.wait_for_messages(
    #     node_wfm, ["/simple_sim_env/init_robot_pose"], [geometry_msgs.msg.PoseStamped],
    #     verbose=True)[0]
    # print(robot_pose_msg)

    node_foo.destroy_node()
    node_bar.destroy_node()
    node_wfm.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    test()
