# ROS Utilities
import math
import numpy as np

import sys
import rospy
import geometry_msgs
import std_msgs
import visualization_msgs
import message_filters

import tf
import tf2_ros
#!!! NEED THIS:
# https://answers.ros.org/question/95791/tf-transformpoint-equivalent-on-tf2/?answer=394789#post-id-394789
# STUPID ROS PROBLEM.
import tf2_geometry_msgs.tf2_geometry_msgs

from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import remap
import sloop_object_search.grpc.common_pb2 as common_pb2
from sloop_object_search.utils.colors import color_map, cmaps


def IS_TAG(t):
    return len(t) == 2 or len(t[2]) == 0

class ROSLaunchWriter:
    """A ROS LAUNCH FILE CONSISTS OF
    'blocks'

    A 'tag' is a special kind of 'block' that
    contains nothing in it.

    Internally, we represent a block as:

    (block_type, <stuff> [, blocks])

    where <stuff> is a dictionary that specifies the options to
    the block (XML options).  And optionally, one can put
    'blocks' at the end which will make those blocks the children
    of the block `block_name`.
    """

    def __init__(self):
        self._blocks = []

    def add_tag(self, tag_type, options):
        """Adds a single, unnested <tag ... /> tag.
        `kwargs` are the options.
        For example, arg.

        options is a dictionary that specifies the XML tag options.

        Note that a tag is a special kind of 'block'"""
        self._blocks.append((tag_type, options))

    def add_block(self, block_type, options, blocks):
        """
        Adds a block. A block looks like:

        <block_type ...>
            <block> ... </block>
            .... [stuff in blocks]
        </block>

        Args:
            blocks (list): list of blocks.
        """
        self._blocks.append((block_name, options, blocks))

    def add_blocks(self, blocks):
        self._blocks.extend(blocks)

    @staticmethod
    def make_block(block_type, options, blocks):
        """Same specification as `add_block` except instead of
        adding the block into self._blocks, returns the block."""
        return (block_type, options, blocks)

    @staticmethod
    def make_tag(tag_type, options):
        """Same specification as `add_tag` except instead of
        adding the block into self._blocks, returns the block."""
        return (tag_type, options)

    def _dump_block(self, block, indent_level, indent_size=4):
        block_type = block[0]
        options = block[1]
        block_str = (" "*(indent_level*indent_size)) + "<" + block_type + " "
        for opt_name in options:
            opt_val = options[opt_name]
            block_str += "{}=\"{}\" ".format(opt_name, opt_val)
        if IS_TAG(block):
            block_str += "\>\n"
        else:
            block_str += ">\n"
            for subblock in block[2]:
                block_str += self._dump_block(subblock, indent_level+1)
            block_str += "\n</{}>\n".format(block_type)
        return block_str

    def dump(self, f=None, indent_size=4):
        """Outputs the roslaunch file to given file stream, if provided.
        Otherwise, returns the entire string of the XML file."""
        lines = "<?xml version=\"1.0\"?>\n"
        lines += "<launch>\n"
        for block in self._blocks:
            lines += self._dump_block(block, 0, indent_size=indent_size) + "\n"
        lines += "</launch>"
        if f is not None:
            f.writelines(lines)
        else:
            return "".join(lines)

### Pose and Transforms ###
def pose_to_tuple(pose):
    """
    Given a geometry_msgs/Pose message,
    returns a tuple (x, y, z, qx, qy, qz, qw)
    """
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    return (x, y, z, qx, qy, qz, qw)

def transform_to_tuple(transform):
    x = transform.translation.x
    y = transform.translation.y
    z = transform.translation.z
    qx = transform.rotation.x
    qy = transform.rotation.y
    qz = transform.rotation.z
    qw = transform.rotation.w
    return (x, y, z, qx, qy, qz, qw)

def transform_to_pose_stamped(transform, frame_id, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()

    pose_msg = geometry_msgs.msg.PoseStamped()
    pose_msg.header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    pose_msg.pose.position = geometry_msgs.msg.Point(x=transform.translation.x,
                                                     y=transform.translation.y,
                                                     z=transform.translation.z)
    pose_msg.pose.orientation = transform.rotation
    return pose_msg

def pose_tuple_to_pose_stamped(pose_tuple, frame_id, stamp=None):
    x, y, z, qx, qy, qz, qw = pose_tuple
    if stamp is None:
        stamp = rospy.Time.now()

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

def topic_exists(topic):
    all_topics = [t[0] for t in rospy.get_published_topics()]
    return topic in all_topics

def joint_state_dict(position, names):
    return {names[i]:position[i]
            for i in range(len(names))}

def tf2_frame_eq(f1, f2):
    if f1[0] == "/":
        f1 = f1[1:]
    if f2[0] == "/":
        f2 = f2[1:]
    return f1 == f2

def tf2_frame(f):
    # Remove leading slash in tf base frame (you can't have leading slashes on TF2 frame names);
    # reference: https://github.com/ros-planning/navigation/issues/794#issuecomment-433465562
    if f[0] == "/":
        f = f[1:]
    return f

def tf2_header(h):
    h.frame_id = tf2_frame(h.frame_id)
    return h

def tf2_transform(tf2buf, object_stamped, target_frame):
    """
    transforms the stamped object into the target frame.
    """
    # remove leading slash in frame if it exists (tf2's requirement)
    object_stamped.header = tf2_header(object_stamped.header)
    result_stamped = None
    try:
        result_stamped = tf2buf.transform(object_stamped, target_frame)
    except:
        einfo = sys.exc_info()
        msg = "{}: {}".format(einfo[0], einfo[1])
        rospy.logerr(msg)
    finally:
        return result_stamped

def tf2_lookup_transform(tf2buf, target_frame, source_frame, timestamp):
    """If timestamp is None, will get the most recent transform"""
    try:
        return tf2buf.lookup_transform(tf2_frame(target_frame),
                                       tf2_frame(source_frame),
                                       timestamp)
    except tf2_ros.LookupException:
        rospy.logerr("Error looking up transform from {} to {}"\
                     .format(target_frame, source_frame))

### Mathematics ###
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def vec_norm(v):
    return math.sqrt(sum(a**2 for a in v))

def quat_diff(q1, q2):
    """returns the quaternion space difference between two
    quaternion q1 and q2"""
    # reference: https://stackoverflow.com/a/22167097/2893053
    # reference: http://wiki.ros.org/tf2/Tutorials/Quaternions#Relative_rotations
    x1, y1, z1, w1 = q1
    q1_inv = (x1, y1, z1, -w1)
    qd = tf.transformations.quaternion_multiply(q2, q1_inv)
    return qd

def quat_diff_angle(q1, q2):
    """returns the angle (radians) between q1 and q2; signed"""
    # reference: https://stackoverflow.com/a/23263233/2893053
    qd = quat_diff(q1, q2)
    return 2*math.atan2(vec_norm(qd[:3]), qd[3])

def quat_diff_angle_relative(q1, q2):
    """returns the angle (radians) between q1 and q2;
    The angle will be the smallest one between the two
    vectors. It is the "intuitive" angular difference.
    Unsigned."""
    # reference: https://stackoverflow.com/a/23263233/2893053
    ad = quat_diff_angle(q1, q2)
    return min(2*math.pi - ad, ad)

def to_degrees(th):
    return th*180 / math.pi

def to_radians(th):
    return th*math.pi / 180

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


### Sensor Messages ###
import sensor_msgs.msg as sensor_msgs
import cv_bridge
def convert(msg_or_img, encoding='passthrough'):
    if isinstance(msg_or_img, sensor_msgs.Image):
        return _convert_imgmsg(msg_or_img, encoding=encoding)
    elif isinstance(msg_or_img, np.ndarray):
        return _convert_img(msg_or_img, encoding=encoding)
    raise ValueError("Cannot handle message type {}".format(msg_or_img))

def _convert_imgmsg(msg, encoding='passthrough'):
    bridge = cv_bridge.CvBridge()
    cv2_image = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
    return cv2_image

def _convert_img(img, encoding='passthrough'):
    bridge = cv_bridge.CvBridge()
    msg = bridge.cv2_to_imgmsg(img, encoding=encoding)
    return msg

### Visualization ###
from visualization_msgs.msg import Marker, MarkerArray
def make_viz_marker_from_object_state(sobj, header, **kwargs):
    """
    Args:
       sobj (ObjectState)
       viz_type (int): e.g. Marker.CUBE
       color (std_msgs.ColorRGBA)
       scale (float or geometry_msgs.Vector3)
    """
    marker = Marker(header=header)
    marker.id = hash16(sobj["id"])
    loc = sobj.loc
    marker.pose.position = geometry_msgs.msg.Point(x=loc[0], y=loc[1], z=loc[2])
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_for_object(objid, pose, header, **kwargs):
    """
    Args:
       pose (x,y,z,qx,qy,qz,qw)
       viz_type (int): e.g. Marker.CUBE
       color (std_msgs.ColorRGBA)
       scale (float or geometry_msgs.Vector3)
    """
    marker = Marker(header=header)
    marker.ns = "object"
    marker.id = hash16(objid)
    loc = pose[:3]
    rot = pose[3:]
    marker.pose.position = geometry_msgs.msg.Point(x=loc[0], y=loc[1], z=loc[2])
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
    _fill_viz_marker(marker, **kwargs)
    return marker

def tf2msg_from_object_loc(loc, world_frame, object_frame, **kwargs):
    stamp = kwargs.get("stamp", rospy.Time.now())
    t = geometry_msgs.msg.TransformStamped(
        header=std_msgs.msg.Header(stamp=stamp,
                                   frame_id=world_frame))
    t.child_frame_id = object_frame
    t.transform.translation = geometry_msgs.msg.Vector3(x=loc[0], y=loc[1], z=loc[2])
    t.transform.rotation = geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    return t

def make_viz_marker_from_robot_state(srobot, header, **kwargs):
    marker = Marker(header=header)
    marker.id = hash16(srobot["id"])
    x,y,z,qx,qy,qz,qw = srobot.pose
    marker.pose.position = geometry_msgs.msg.Point(x=x, y=y, z=z)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    kwargs["viz_type"] = Marker.ARROW
    _fill_viz_marker(marker, **kwargs)
    return marker

def tf2msg_from_robot_pose(robot_pose, world_frame, robot_frame, **kwargs):
    stamp = kwargs.get("stamp", rospy.Time.now())
    t = geometry_msgs.msg.TransformStamped(
        header=std_msgs.msg.Header(stamp=stamp,
                                   frame_id=world_frame))
    t.child_frame_id = robot_frame
    x,y,z,qx,qy,qz,qw = robot_pose
    t.transform.translation = geometry_msgs.msg.Vector3(x=x, y=y, z=z)
    t.transform.rotation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    return t

def make_viz_marker_from_robot_pose_3d(robot_id, robot_pose, header, **kwargs):
    marker = Marker(header=header)
    marker.id = hash16(robot_id)
    x,y,z,qx,qy,qz,qw = robot_pose
    marker.pose.position = geometry_msgs.msg.Point(x=x, y=y, z=z)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    kwargs["viz_type"] = Marker.ARROW
    _fill_viz_marker(marker, **kwargs)
    return marker

def _fill_viz_marker(marker, action=Marker.ADD, viz_type=Marker.CUBE,
                     color=[0.0, 0.8, 0.0, 0.8], scale=1.0, lifetime=1.0):
    marker.type = viz_type
    if type(scale) == float:
        marker.scale = geometry_msgs.msg.Vector3(x=scale, y=scale, z=scale)  # we don't care about scale
    else:
        marker.scale = scale  # we don't care about scale
    marker.action = action
    marker.lifetime = rospy.Duration(lifetime)
    color = std_msgs.msg.ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    marker.color = color

def make_octnode_marker_msg(pos, res, header, alpha=1.0,
                            lifetime=1.0, color=[0.0, 0.8, 0.0]):
    """
    Creates an rviz marker for a OctNode, specified
    by the given 3D position (in frame of header),
    resolution (in meters), and with transparency determined by
    given probability.
    """
    marker = Marker()
    marker.header = header
    marker.ns = "octnode"
    marker.id = hash16((*pos, res))
    marker.type = Marker.CUBE
    marker.pose.position = geometry_msgs.msg.Point(x=pos[0] + res/2,
                                                   y=pos[1] + res/2,
                                                   z=pos[2] + res/2)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    marker.scale = geometry_msgs.msg.Vector3(x=res, y=res, z=res)
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(lifetime)
    marker.color = std_msgs.msg.ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
    return marker

def clear_markers(header, ns):
    marker = Marker()
    marker.header = header
    marker.ns = ns
    marker.action = Marker.DELETEALL
    return MarkerArray([marker])

def make_octree_belief_proto_markers_msg(octree_belief_pb, header, cmap=cmaps.COLOR_MAP_JET,
                                         alpha_scaling=1.0):
    """given an octree belief's protobuf representation,
    which is a Histogram, make a MarkerArray message for it."""
    def _compute_alpha(p, vmin, vmax):
        if vmax - vmin > 0.0:
            return remap(p, vmin, vmax, 0.0, 0.8)
        else:
            return 0.8

    markers = []
    hist_pb = octree_belief_pb.dist
    prob_max = max(hist_pb.probs)
    prob_min = min(hist_pb.probs)
    for i in range(hist_pb.length):
        voxel = common_pb2.Voxel3D()
        hist_pb.values[i].Unpack(voxel)

        pos = [voxel.pos.x, voxel.pos.y, voxel.pos.z]
        prob = hist_pb.probs[i]
        color = color_map(prob, [prob_min, prob_max], cmap)
        alpha = _compute_alpha(prob, prob_min, prob_max) * alpha_scaling
        marker = make_octnode_marker_msg(
            pos, voxel.res, header, lifetime=0,  # 0 is forever
            color=color, alpha=alpha)
        markers.append(marker)
    return MarkerArray(markers)


### Communication ###
class WaitForMessages:
    """deals with waiting for messages to arrive at multiple
    topics. Uses ApproximateTimeSynchronizer. Simply returns
    a tuple of messages that were received."""
    def __init__(self, topics, mtypes, queue_size=10,
                 delay=0.2, allow_headerless=False, sleep=0.5, verbose=False):
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
        """
        self.messages = None
        self.verbose = verbose
        if self.verbose:
            rospy.loginfo("initializing message filter ApproximateTimeSynchronizer")
        self.subs = [message_filters.Subscriber(topic, mtype)
                     for topic, mtype in zip(topics, mtypes)]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subs, queue_size, delay, allow_headerless=allow_headerless)
        self.ts.registerCallback(self._cb)
        rate = rospy.Rate(1.0/sleep)
        while not rospy.is_shutdown():
            if self.messages is not None:
                rospy.loginfo("Received messages! Done!")
                break
            if self.verbose:
                rospy.loginfo("waiting for messages from {}".format(topics))
            rate.sleep()

    def _cb(self, *messages):
        if self.messages is not None:
            return
        if self.verbose:
            rospy.loginfo("got messages!")
        self.messages = messages
