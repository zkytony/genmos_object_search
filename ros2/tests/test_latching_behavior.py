# This script illustrates the basic behavior of
# latching in ROS2
#
# Suppose:
#   pub -> /test (latches)
#            \---> sub
import rclpy
import threading
import std_msgs.msg
from genmos_ros2 import ros2_utils

def cb(msg):
    print(f"MESSAGE: {msg.data}")

def test():
    rclpy.init()
    node1 = rclpy.create_node("pub")
    pub = node1.create_publisher(std_msgs.msg.String, "test",
                                 ros2_utils.latch(depth=10))
    node2 = rclpy.create_node("sub")
    executor = rclpy.executors.MultiThreadedExecutor(3)
    executor.add_node(node1)
    executor.add_node(node2)
    pub.publish(std_msgs.msg.String(data="hello world"))
    t_ex = threading.Thread(target=executor.spin, args=(), daemon=False)
    t_ex.start()
    # Different from ROS1, a new subscriber always consumes the most
    # recent message from a latched topic.  So let's say topic '/test'
    # is latched, and the user calls wait_for_messages(['/foo'], ...).
    # For the first time, the user should get the most recent message
    # from '/foo'. But for the second time, the user ALSO gets the same
    # message from '/foo'.  This is because every call creates new
    # mesage_filter Subscriber objects -- this is also reasonable
    # because one could expect every time I subscribe to a latched
    # topic, I should get the most recent message.
    #
    # So, we expect the following two calls to both return and the
    # second one return quickly with the same msg.
    msg = ros2_utils.wait_for_messages(
        node2, ["test"], [std_msgs.msg.String],
        allow_headerless=True, verbose=True,
        latched_topics={"test"})[0]
    print(msg.data)
    msg = ros2_utils.wait_for_messages(
        node2, ["test"], [std_msgs.msg.String],
        allow_headerless=True, verbose=True,
        latched_topics={"test"})[0]
    print(msg.data)
    t_ex.join()

    node1.destroy_node()
    node2.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test()
