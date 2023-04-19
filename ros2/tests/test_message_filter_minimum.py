# Using ROS Humble
# /author: Kaiyu Zheng
import rclpy
import message_filters
import geometry_msgs.msg
import std_msgs.msg
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


class NodeFoo(Node):
    # publishing is latched
    def __init__(self):
        super().__init__("foo")

        point_foo = geometry_msgs.msg.PointStamped()
        point_foo.header = std_msgs.msg.Header(stamp=self.get_clock().now().to_msg())
        point_foo.point.x = 1.0
        point_foo.point.y = 0.0
        point_foo.point.z = 0.0

        self.pub = self.create_publisher(
            geometry_msgs.msg.PointStamped, "foo_topic",
            QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.pub.publish(point_foo)
        self.get_logger().info("foo published!")


class NodeBar(Node):
    def __init__(self):
        super().__init__("bar")

        point_bar = geometry_msgs.msg.PointStamped()
        point_bar.header = std_msgs.msg.Header(stamp=self.get_clock().now().to_msg())
        point_bar.point.x = 0.0
        point_bar.point.y = 0.0
        point_bar.point.z = 1.0
        self.point_bar = point_bar

        self.pub = self.create_publisher(geometry_msgs.msg.PointStamped, "bar_topic", 10)
        self.timer = self.create_timer(0.5, self.publish_msg)

    def publish_msg(self):
        self.point_bar.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.point_bar)
        self.get_logger().info("bar published!")


class Syncer(Node):
    def __init__(self):
        super().__init__("syncer")
        sub_foo = message_filters.Subscriber(
            self, geometry_msgs.msg.PointStamped, "foo_topic",
            qos_profile=QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        sub_bar = message_filters.Subscriber(
            self, geometry_msgs.msg.PointStamped, "bar_topic")

        queue_size = 10
        delay = 1.0
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_foo, sub_bar], queue_size, delay)
        self.ts.registerCallback(self._cb)

    def _cb(self, msg_foo, msg_bar):
        self.get_logger().info("Received messages!!")
        self.get_logger().info(f"from Foo: {msg_foo}")
        self.get_logger().info(f"from Bar: {msg_bar}")

def test():
    rclpy.init()
    node_foo = NodeFoo()
    node_bar = NodeBar()
    syncer = Syncer()

    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(node_foo)
    executor.add_node(node_bar)
    executor.add_node(syncer)
    executor.spin()

    node_foo.destroy_node()
    node_bar.destroy_node()
    syncer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    test()
