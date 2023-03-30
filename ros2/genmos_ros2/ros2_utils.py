from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import geometry_msgs.msg
import std_msgs.msg

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
