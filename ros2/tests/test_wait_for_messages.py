# To run this test, run
#
# ros2 topic pub "test1" std_msgs/msg/String "data: 'hello'
# ros2 topic pub "test2" std_msgs/msg/String "data: 'hello'
import threading

import rclpy
from rclpy.node import Node
import std_msgs.msg

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

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node_foo)
    executor.add_node(node_bar)

    t_ex = threading.Thread(target=executor.spin, args=(), daemon=False)
    t_ex.start()

    print(ros2_utils.wait_for_messages(
        ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2, latched_topics={"/test1"}))
    print(ros2_utils.wait_for_messages(
        ["/test1", "/test2"], [std_msgs.msg.String, std_msgs.msg.String],
        allow_headerless=True, verbose=True, delay=0.5,
        timeout=2, latched_topics={"/test1"}))

    node_foo.destroy_node()
    node_bar.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    test()
