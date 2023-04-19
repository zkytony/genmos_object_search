# Let's say we'd like to have a node that
#
# 1. publishes to a topic 'state' periodically (based on an internally
# maintained state)
#
# 2. subscribes to a topic 'action'; each received action message causes the
# state to change, a process that takes several seconds to complete.
#
# Question: How could we implement this node such that the action-change process
# does not block state publishing?
# /author: Kaiyu Zheng
import time
import rclpy
import geometry_msgs.msg
import std_msgs.msg
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


class NodeSol(Node):
    """This is the node we desire"""
    def __init__(self):
        super().__init__("node_sol")
        self._state = geometry_msgs.msg.PointStamped()
        self._state.header = std_msgs.msg.Header(stamp=self.get_clock().now().to_msg())
        self._state.point.x = 1.0
        self._state.point.y = 0.0
        self._state.point.z = 0.0
        self.state_pub = self.create_publisher(
            geometry_msgs.msg.PointStamped, "state",
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(0.3, self.publish_state)

    def publish_state(self):
        self._state.header.stamp = self.get_clock().now().to_msg()
        self.state_pub.publish(self._state)
        self.get_logger().info("state published! ({}, {}, {})"\
                               .format(self._state.point.x,
                                       self._state.point.y,
                                       self._state.point.z))

def test():
    rclpy.init()
    node_sol = NodeSol()
    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(node_sol)
    executor.spin()

    node_foo.destroy_node()
    node_bar.destroy_node()
    syncer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    test()
