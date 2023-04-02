import random
import uuid
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.executors import SingleThreadedExecutor

import message_filters
import geometry_msgs.msg
import std_msgs.msg

from genmos_object_search import utils


def print_parameters(node, names):
    rclparams = node.get_parameters(names)
    for rclparam, name in zip(rclparams, names):
        node.get_logger().info(f"- {name}: {rclparam.value}")

def latch(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL):
    return QoSProfile(depth=depth, durability=durability)

class WrappedNode(Node):
    def __init__(self, node_name, params=None, verbose=True):
        """
        The Wrapped ROS2 Node.

        Args:
            node_name (str): name of node
            params (list): list of (parameter name, default value) tuples.
        """
        super().__init__(node_name)
        self._param_names = set()
        if params is None:
            params = []
        for param_name, default_value in params:
            self.declare_parameter(param_name, default_value)
            self._param_names.add(param_name)

        # print parameters on start
        if verbose:
            self.log_info("Initializing node {}. Parameters:".format(self.get_name()))
            print_parameters(self, self._param_names)

    def log_info(self, note):
        self.get_logger().info(note)


def pose_tuple_to_pose_stamped(pose_tuple, frame_id, stamp=None, node=None):
    x, y, z, qx, qy, qz, qw = pose_tuple
    if stamp is None:
        if node is not None:
            stamp = node.get_clock().now().to_msg()

    pose_msg = geometry_msgs.msg.PoseStamped()
    pose_msg.header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    pose_msg.pose.position = geometry_msgs.msg.Point(x=x,
                                                     y=y,
                                                     z=z)
    pose_msg.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose_msg

def pose_tuple_from_pose_stamped(pose_stamped_msg):
    position = pose_stamped_msg.pose.position
    orientation = pose_stamped_msg.pose.orientation
    return (position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w)

### Communication ###
class WaitForMessagesNode(Node):
    """deals with waiting for messages to arrive at multiple
    topics. Uses ApproximateTimeSynchronizer. Simply returns
    a tuple of messages that were received."""
    def __init__(self, topics, mtypes, queue_size=10, delay=0.2,
                 allow_headerless=False, sleep=0.5, timeout=None,
                 verbose=False, exception_on_timeout=False):
        """
        Args:
            topics (list) List of topics
            mtypes (list) List of message types, one for each topic.
            delay  (float) The delay in seconds for which the messages
                could be synchronized.
            allow_headerless (bool): Whether it's ok for there to be
                no header in the messages.
            sleep (float) the amount of time to wait before checking
                whether messages are received
            timeout (float or None): Time in seconds to wait. None if forever.
                If exceeded timeout, self.messages will contain None for
                each topic.
        """
        super().__init__('wait_for_messages')
        self.messages = None
        self.verbose = verbose
        self.topics = topics
        self.timeout = timeout
        self.exception_on_timeout = exception_on_timeout
        self.has_timed_out = False

        if self.verbose:
            self.get_logger().info("initializing message filter ApproximateTimeSynchronizer")
        self.subs = [message_filters.Subscriber(self, mtype, topic)
                     for topic, mtype in zip(topics, mtypes)]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subs, queue_size, delay, allow_headerless=allow_headerless)
        self.ts.registerCallback(self._cb)

        self._start_time = self.get_clock().now()
        self.timer = self.create_timer(sleep, self.check_messages_received)

    def check_messages_received(self):
        if self.messages is not None:
            self.get_logger().info("WaitForMessages: Received messages! Done!")
            return
        if self.verbose:
            self.get_logger().info("WaitForMessages: waiting for messages from {}".format(self.topics))
        _dt = self.get_clock().now() - self._start_time
        if self.timeout is not None and _dt.nanoseconds*1e-9 > self.timeout:
            self.get_logger().error("WaitForMessages: timeout waiting for messages")
            self.messages = [None]*len(self.topics)
            self.has_timed_out = True
            if self.exception_on_timeout:
                raise TimeoutError("WaitForMessages: timeout waiting for messages")
            return

    def _cb(self, *messages):
        if self.messages is not None:
            return
        if self.verbose:
            self.get_logger().info("WaitForMessages: got messages!")
        self.messages = messages


def wait_for_messages(*args, **kwargs):
    """A wrapper for running WaitForMessagesNode."""
    wfm_node = WaitForMessagesNode(*args, **kwargs)
    try:
        while wfm_node.messages is None and not wfm_node.has_timed_out:
            rclpy.spin_once(wfm_node)
    except TimeoutError as ex:
        raise ex
    finally:
        wfm_node.destroy_node()
    return wfm_node.messages
