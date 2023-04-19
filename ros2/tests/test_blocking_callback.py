# Let's say we'd like to have a node that
#
# 1. publishes to a topic 'state' periodically (based on an internally
# maintained state)
#
# 2. subscribes to a topic 'action'; each received action message causes the
# state to change, a process that takes several seconds to complete.
#
# 3. publishes to a topic 'action_done' when the action execution has finished.
#    (does not execute an action if a previous action was executing already).
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
            geometry_msgs.msg.PointStamped, "~/state",
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.action_sub = self.create_subscription(
            std_msgs.msg.String, "~/action", self.action_cb,
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.action_done_pub = self.create_publisher(
            std_msgs.msg.String, "~/action_done",
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(0.1, self.publish_state)
        self._executing = False

    def publish_state(self):
        self._state.header.stamp = self.get_clock().now().to_msg()
        self.state_pub.publish(self._state)
        self.get_logger().info("state published! ({:.3f}, {:.3f}, {:.3f})"\
                               .format(self._state.point.x,
                                       self._state.point.y,
                                       self._state.point.z))

    def action_cb(self, msg):
        if self._executing:
            self.get_logger().info("action ignored - another action is executing")
        else:
            self.get_logger().info("action received!")
            self._executing = True
            # self.execute_action(msg.data)
            self.action_done_pub.publish(std_msgs.msg.String(data="done"))
            self._executing = False

    def execute_action(self, action):
        """changes the state according to the action -- the action doesn't
        matter; the point is, this function will take some time to finish.
        """
        for i in range(10):
            self._state.point.x += 0.1
            time.sleep(0.1)



class NodeActionPub(Node):
    """a node used to publish actions for testing;
    """
    def __init__(self):
        super().__init__("node_action_pub")
        self.action_pub = self.create_publisher(
            std_msgs.msg.String, "/node_sol/action",
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(1.0, self.publish_action)

    def publish_action(self):
        self.action_pub.publish(std_msgs.msg.String(data="action"))


def test():
    rclpy.init()
    node_sol = NodeSol()
    node_action_pub = NodeActionPub()
    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(node_sol)
    executor.add_node(node_action_pub)
    executor.spin()

    node_foo.destroy_node()
    node_bar.destroy_node()
    syncer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    test()
