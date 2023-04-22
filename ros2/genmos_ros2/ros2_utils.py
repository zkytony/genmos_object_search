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
import vision_msgs.msg
import ros2_numpy
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs

import google.protobuf.timestamp_pb2
import genmos_object_search.grpc.common_pb2 as common_pb2
import genmos_object_search.grpc.observation_pb2 as o_pb2
import genmos_object_search.grpc.common_pb2 as c_pb2
from genmos_object_search import utils
from genmos_object_search.grpc.utils import proto_utils
from genmos_object_search.utils import math as math_utils
from genmos_object_search.utils.misc import hash16
from genmos_object_search.utils.colors import color_map, cmaps, lighter_with_alpha


### Logging ###
from rclpy.impl import rcutils_logger
LOGGER_NAME = "ros2_utils"
RCUTILS_LOGGER = rcutils_logger.RcutilsLogger(name=LOGGER_NAME)

def log_info(msg):
    """msg (str): the log content"""
    RCUTILS_LOGGER.info(msg)

def log_error(msg):
    """msg (str): the log content"""
    RCUTILS_LOGGER.error(msg)


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

def transform_to_pose_stamped(transform, frame_id, stamp=None, node=None):
    if stamp is None:
        if node is not None:
            stamp = node.get_clock().now().to_msg()

    pose_msg = geometry_msgs.msg.PoseStamped()
    pose_msg.header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    pose_msg.pose.position = geometry_msgs.msg.Point(x=transform.translation.x,
                                                     y=transform.translation.y,
                                                     z=transform.translation.z)
    pose_msg.pose.orientation = transform.rotation
    return pose_msg


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


### Node ###
def declare_params(node, params):
    """params (list): list of (parameter name, default value) tuples."""
    for param_name, default_value in params:
        node.declare_parameter(param_name, default_value)

def print_parameters(node, names):
    """
    Args:
        node (Node): the node for which we care about the parameter values
        names (list): either a list of [param_name ... ] or [(param_name, default_value)...]
    """
    if len(names) > 0:
        if hasattr(names[0], "__len__"):
            if len(names[0]) != 2:
                raise TypeError("expecting 'names' to be [param_name ... ] or [(param_name, default_value)...]")
            else:
                names = [p[0] for p in names]
    rclparams = node.get_parameters(names)
    for rclparam, name in zip(rclparams, names):
        node.get_logger().info(f"- {name}: {rclparam.value}")

def latch(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL):
    return QoSProfile(depth=depth, durability=durability)

### vision_msgs ###
def make_bbox3d_msg(center, sizes):
    if len(center) == 7:
        x, y, z, qx, qy, qz, qw = center
        q = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    else:
        x, y, z = center
        q = geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    s1, s2, s3 = sizes
    msg = vision_msgs.msg.BoundingBox3D()
    msg.center.position = geometry_msgs.msg.Point(x=x, y=y, z=z)
    msg.center.orientation = q
    msg.size = geometry_msgs.msg.Vector3(x=s1, y=s2, z=s3)
    return msg



### Communication ###
def wait_for_messages(node, topics, mtypes, **kwargs):
    """Waits for messages to arrive at multiple topics within a given
    time window. Uses message_filters.ApproximateTimeSynchronizer.
    This function blocks until receiving the messages or when a given
    timeout expires. Assumes the given node is spinning by
    some external executor.

    Requires the user to pass in a node, since
    message_filters.Subscriber requires a node upon construction.

    Args:
        node (rclpy.Node): the node being attached
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
        latched_topics (set): a set of topics for which the publisher latches (i.e.
            sets QoS durability to transient_local).
    """
    return _WaitForMessages(node, topics, mtypes, **kwargs).messages

class _WaitForMessages:
    def __init__(self, node, topics, mtypes, queue_size=10, delay=0.2,
                 allow_headerless=False, sleep=0.5, timeout=None,
                 verbose=False, exception_on_timeout=False,
                 latched_topics=None, callback_group=None):
        self.node = node
        self.messages = None
        self.verbose = verbose
        self.topics = topics
        self.timeout = timeout
        self.exception_on_timeout = exception_on_timeout
        self.has_timed_out = False
        if latched_topics is None:
            latched_topics = set()
        self.latched_topics = latched_topics

        if self.verbose:
            log_info("initializing message filter ApproximateTimeSynchronizer")
        self.subs = [self._message_filters_subscriber(mtype, topic, callback_group=callback_group)
                     for topic, mtype in zip(topics, mtypes)]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subs, queue_size, delay, allow_headerless=allow_headerless)
        self.ts.registerCallback(self._cb)

        self._start_time = self.node.get_clock().now()
        rate = self.node.create_rate(1./sleep)
        while self.messages is None and not self.has_timed_out:
            if self.check_messages_received():
                break
            rate.sleep()

    def _message_filters_subscriber(self, mtype, topic, callback_group=None):
        if topic in self.latched_topics:
            return message_filters.Subscriber(
                self.node, mtype, topic,
                qos_profile=QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
                callback_group=callback_group)
        else:
            return message_filters.Subscriber(self.node, mtype, topic,
                                              callback_group=callback_group)

    def check_messages_received(self):
        if self.messages is not None:
            log_info("WaitForMessages: Received messages! Done!")
            return True
        if self.verbose:
            log_info("WaitForMessages: waiting for messages from {}".format(self.topics))
        _dt = self.node.get_clock().now() - self._start_time
        if self.timeout is not None and _dt.nanoseconds*1e-9 > self.timeout:
            log_error("WaitForMessages: timeout waiting for messages")
            self.messages = [None]*len(self.topics)
            self.has_timed_out = True
            if self.exception_on_timeout:
                raise TimeoutError("WaitForMessages: timeout waiting for messages")
        return False

    def _cb(self, *messages):
        if self.messages is not None:
            return
        if self.verbose:
            log_info("WaitForMessages: callback got messages!")
        self.messages = messages


### TF2 ###
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
        log_error(msg)
    finally:
        return result_stamped

def tf2_lookup_transform(tf2buf, target_frame, source_frame, timestamp):
    """If timestamp is None, will get the most recent transform"""
    try:
        return tf2buf.lookup_transform(tf2_frame(target_frame),
                                       tf2_frame(source_frame),
                                       timestamp)
    except tf2_ros.LookupException:
        log_error("Error looking up transform from {} to {}"\
                  .format(target_frame, source_frame))

def tf2_do_transform(pose_stamped, trans):
    return tf2_geometry_msgs.do_transform_pose(pose_stamped, trans)


### Visualization ###
from visualization_msgs.msg import Marker, MarkerArray
def make_viz_marker_for_object(objid, pose, header, **kwargs):
    """
    Args:
       pose (x,y,z,qx,qy,qz,qw) or (x,y,z)
       viz_type (int): e.g. Marker.CUBE
       color (std_msgs.ColorRGBA)
       scale (float or geometry_msgs.Vector3)
    """
    marker = Marker(header=header)
    marker.ns = "object"
    _id = kwargs.pop("id", objid)
    marker.id = hash16((_id, pose))
    loc = pose[:3]
    rot = pose[3:]
    marker.pose.position = geometry_msgs.msg.Point(x=loc[0], y=loc[1], z=loc[2])
    if len(rot) > 0:
        marker.pose.orientation = geometry_msgs.msg.Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
    elif len(rot) != 0:
        raise ValueError("rotation in object pose is invalid. "\
                         "Should be quaternion qx, qy, qz, qw or nothing.")
    _fill_viz_marker(marker, **kwargs)
    return marker

def tf2msg_from_robot_pose(robot_pose, world_frame, robot_frame, stamp=None, node=None, **kwargs):
    if stamp is None:
        if node is not None:
            stamp = node.get_clock().now().to_msg()
    t = geometry_msgs.msg.TransformStamped(
        header=std_msgs.msg.Header(stamp=stamp,
                                   frame_id=world_frame))
    t.child_frame_id = robot_frame
    x,y,z,qx,qy,qz,qw = robot_pose
    t.transform.translation = geometry_msgs.msg.Vector3(x=x, y=y, z=z)
    t.transform.rotation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    return t

def tf2msg_from_object_loc(loc, world_frame, object_frame, stamp=None, node=None, **kwargs):
    if stamp is None:
        if node is not None:
            stamp = node.get_clock().now().to_msg()
    t = geometry_msgs.msg.TransformStamped(
        header=std_msgs.msg.Header(stamp=stamp,
                                   frame_id=world_frame))
    t.child_frame_id = object_frame
    t.transform.translation = geometry_msgs.msg.Vector3(x=loc[0], y=loc[1], z=loc[2])
    t.transform.rotation = geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return t

def make_viz_marker_from_robot_pose_3d(robot_id, robot_pose, header, **kwargs):
    marker = Marker(header=header)
    marker.id = hash16(robot_id)
    x,y,z,qx,qy,qz,qw = robot_pose
    marker.pose.position = geometry_msgs.msg.Point(x=float(x), y=float(y), z=float(z))
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    kwargs["viz_type"] = Marker.ARROW
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_for_line_segment(start_point, end_point, header, **kwargs):
    marker = Marker(header=header)
    _id = kwargs.pop("id", hash16((start_point, end_point)))
    marker.id = _id
    marker.points = [geometry_msgs.msg.Point(**unravel_args(["x","y","z"], start_point)),
                     geometry_msgs.msg.Point(**unravel_args(["x","y","z"], end_point))]
    marker.pose.position = geometry_msgs.msg.Point(x=0.0, y=0.0, z=0.0)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    kwargs["viz_type"] = Marker.LINE_STRIP
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_cylinder(_id, pose, header, **kwargs):
    """generic method for cylinder; note that pose should be a 7-d tuple"""
    marker = Marker(header=header)
    marker.id = _id
    x, y, z, qx ,qy ,qz, qw = pose
    marker.pose.position = geometry_msgs.msg.Point(
        x=float(x),y=float(y),z=float(z))
    marker.pose.orientation = geometry_msgs.msg.Quaternion(
        x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    kwargs["viz_type"] = Marker.CYLINDER
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_cube(_id, pose, header, **kwargs):
    """generic method for cylinder; note that pose should be a 7-d tuple"""
    marker = Marker(header=header)
    marker.id = _id
    x, y, z, qx ,qy ,qz, qw = pose
    marker.pose.position = geometry_msgs.msg.Point(
        x=float(x),y=float(y),z=float(z))
    marker.pose.orientation = geometry_msgs.msg.Quaternion(
        x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    kwargs["viz_type"] = Marker.CUBE
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
    marker.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()
    color = std_msgs.msg.ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    marker.color = color

def make_viz_marker_for_voxel(objid, voxel, header, ns="voxel", **kwargs):
    """voxel is (x,y,z,r), where x,y,z is a point in world frame and
    r is the size in meters."""
    return make_octnode_marker_msg(objid, voxel[:3], voxel[3], header, ns=ns, **kwargs)

def make_octnode_marker_msg(objid, pos, res, header, alpha=1.0,
                            lifetime=1.0, color=[0.0, 0.8, 0.0], ns="octnode"):
    """
    Creates an rviz marker for a OctNode, specified
    by the given 3D position (in frame of header),
    resolution (in meters), and with transparency determined by
    given probability.
    """
    marker = Marker()
    marker.header = header
    marker.ns = ns
    marker.id = hash16((objid, *pos, res))
    marker.type = Marker.CUBE
    marker.pose.position = geometry_msgs.msg.Point(x=pos[0] + res/2,
                                                   y=pos[1] + res/2,
                                                   z=pos[2] + res/2)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    marker.scale = geometry_msgs.msg.Vector3(x=res, y=res, z=res)
    marker.action = Marker.ADD
    marker.lifetime = rclpy.duration.Duration(seconds=lifetime).to_msg()
    marker.color = std_msgs.msg.ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
    return marker

def clear_markers(header, ns):
    """BUG: RVIZ actually clears all messages"""
    marker = Marker()
    marker.header = header
    marker.ns = ns
    marker.action = Marker.DELETEALL
    return MarkerArray(markers=[marker])

def _compute_alpha(p, vmin, vmax):
    if vmax - vmin > 0.0:
        return math_utils.remap(p, vmin, vmax, 0.001, 0.7)
    else:
        return 0.7

def make_octree_belief_proto_markers_msg(octree_belief_pb, header, cmap=cmaps.COLOR_MAP_JET,
                                         alpha_scaling=1.0, prob_thres=None):
    """given an octree belief's protobuf representation,
    which is a Histogram, make a MarkerArray message for it."""
    markers = []
    hist_pb = octree_belief_pb.dist
    probs = []
    voxels = []
    for i in range(hist_pb.length):
        voxel = common_pb2.Voxel3D()
        hist_pb.values[i].Unpack(voxel)
        prob = hist_pb.probs[i] / (voxel.res**3)
        probs.append(prob)
        voxels.append(voxel)
    prob_max = max(probs)
    prob_min = min(probs)
    if prob_min == prob_max:
        prob_min -= 1e-12  # avoid nan
    for i in range(hist_pb.length):
        voxel = voxels[i]
        pos = [voxel.pos.x, voxel.pos.y, voxel.pos.z]
        prob = probs[i]
        if prob_thres is not None:
            if prob < prob_thres:
                continue
        color = color_map(prob, [prob_min, prob_max], cmap)
        alpha = _compute_alpha(prob, prob_min, prob_max) * alpha_scaling
        marker = make_octnode_marker_msg(
            octree_belief_pb.object_id, pos, voxel.res, header, lifetime=0,  # 0 is forever
            color=color, alpha=alpha)
        markers.append(marker)
    return MarkerArray(markers=markers)

def make_object_belief2d_proto_markers_msg(object_belief2d_pb, header,
                                           search_space_resolution,
                                           color=[0.2, 0.7, 0.2],
                                           alpha_scaling=1.0, pos_z=0,
                                           alpha_max=0.7):
    """search_space_resolution: should be the size of a grid cell"""
    markers = []
    object_id = object_belief2d_pb.object_id
    hist_pb = object_belief2d_pb.dist
    hist = {}
    prob_max = max(hist_pb.probs)
    prob_min = min(hist_pb.probs)
    for i in range(hist_pb.length):
        pos_pb = common_pb2.Vec2()
        hist_pb.values[i].Unpack(pos_pb)
        pos = (pos_pb.x, pos_pb.y, pos_z)
        prob = hist_pb.probs[i]
        hist[pos] = prob

    last_val = -1
    if len(color) == 3:
        color = [*color, alpha_max]
    i = 0
    for pos in reversed(sorted(hist, key=hist.get)):
        if last_val != -1:
            color = lighter_with_alpha(np.asarray(color)*255, 1-hist[pos]/last_val)/255

        stop = color[3] < 0.1
        if not stop:
            marker = make_viz_marker_for_object(object_id, (*pos, 0, 0, 0, 1),
                                                header, id=i,
                                                lifetime=0, color=color,
                                                viz_type=Marker.CYLINDER,
                                                scale=geometry_msgs.msg.Vector3(x=search_space_resolution,
                                                                                y=search_space_resolution,
                                                                                z=0.05))
            markers.append(marker)
            last_val = hist[pos]
            i += 1
            if last_val <= 0:
                break
    return MarkerArray(markers=markers)

def make_topo_map_proto_markers_msg(topo_map_pb, header,
                                    search_space_resolution,
                                    node_color=[0.93, 0.85, 0.1, 0.8],
                                    edge_color=[0.05, 0.65, 0.81, 0.8],
                                    edge_thickness=0.05,
                                    pos_z=0.1,
                                    node_thickness=0.05):
    """If positions of topo map nodes are 2D, then the z coordinate
    of the marker will be set to 'pos_z'."""
    # First, add edge markers and collect nodes
    markers = []
    node_pbs = {}
    for edge_pb in topo_map_pb.edges:
        nid1 = edge_pb.node1.id
        if nid1 not in node_pbs:
            node_pbs[nid1] = edge_pb.node1
        nid2 = edge_pb.node2.id
        if nid2 not in node_pbs:
            node_pbs[nid2] = edge_pb.node2
        # add marker for edge as a Line Strip
        pos1 = proto_utils.pos_from_topo_node(edge_pb.node1)
        pos2 = proto_utils.pos_from_topo_node(edge_pb.node2)
        if len(pos1) == 2:
            pos1 = (*pos1, pos_z)
        if len(pos2) == 2:
            pos2 = (*pos2, pos_z)
        edge_scale = geometry_msgs.msg.Vector3(x=edge_thickness)
        edge_marker = make_viz_marker_for_line_segment(
            pos1, pos2, header, color=edge_color, scale=edge_thickness,
            lifetime=0, id=int(edge_pb.id))
        markers.append(edge_marker)

    for nid in node_pbs:
        node_pb = node_pbs[nid]
        pos = proto_utils.pos_from_topo_node(node_pb)
        if len(pos) == 2:
            pos = (*pos, pos_z)
        pose = (*pos, 0, 0, 0, 1)
        node_scale = geometry_msgs.msg.Vector3(
            x=search_space_resolution*1.5,
            y=search_space_resolution*1.5,
            z=node_thickness)
        node_marker = make_viz_marker_cylinder(
            int(nid), pose, header, color=node_color,
            lifetime=0, scale=node_scale)
        markers.append(node_marker)
    return MarkerArray(markers=markers)


def viz_msgs_for_robot_pose(robot_pose, world_frame, robot_frame, stamp=None, node=None,
                            **kwargs):
    """Given a robot pose in the world frame obtained from SLOOP,
    return a tuple (marker, tf2msg) for visualization. Note that
    this function accounts for the differences in default look
    direction between ROS and SLOOP."""
    if stamp is None:
        if node is not None:
            stamp = node.get_clock().now().to_msg()
    scale = kwargs.pop("scale", geometry_msgs.msg.Vector3(x=0.4, y=0.05, z=0.05))
    lifetime = kwargs.pop("lifetime", 0)

    # The camera in POMDP by default looks at +x (see
    # DEFAULT_3DCAMERA_LOOK_DIRECTION) which matches the default forward
    # direction of arrow marker in RVIZ. That is, in both frames, 0 degree means
    # looking at +x. Therefore, we don't need to rotate the marker.
    header = std_msgs.msg.Header(stamp=stamp, frame_id=robot_frame)
    marker = make_viz_marker_from_robot_pose_3d(
        robot_frame, (0,0,0,*math_utils.euler_to_quat(0, 0, 0)),
        header=header, scale=scale, lifetime=lifetime, **kwargs)
    tf2msg = tf2msg_from_robot_pose(robot_pose, world_frame, robot_frame, stamp=stamp)
    return marker, tf2msg


def viz_msgs_for_robot_pose_proto(robot_pose_proto, world_frame, robot_frame, stamp=None, node=None,
                                  **kwargs):
    """Given a robot_pose_proto obtained from sloop grpc server,
    return a tuple (marker, tf2msg) for visualization. Note that
    this function accounts for the differences in default look
    direction between ROS and SLOOP."""
    robot_pose = proto_utils.robot_pose_from_proto(robot_pose_proto)
    return viz_msgs_for_robot_pose(robot_pose, world_frame, robot_frame, stamp=stamp, node=node, **kwargs)


### Sensor messages ###
def pointcloud2_to_pointcloudproto(cloud_msg):
    """
    Converts a PointCloud2 message to a PointCloud proto message.

    Args:
       cloud_msg (sensor_msgs.PointCloud2)
    """
    pcl_raw_array = ros2_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
    points_xyz_array = ros2_numpy.point_cloud2.get_xyz_points(pcl_raw_array)

    points_pb = []
    for p in points_xyz_array:
        point_pb = o_pb2.PointCloud.Point(pos=c_pb2.Vec3(x=p[0], y=p[1], z=p[2]))
        points_pb.append(point_pb)

    header = c_pb2.Header(stamp=google.protobuf.timestamp_pb2.Timestamp().GetCurrentTime(),
                          frame_id=cloud_msg.header.frame_id)
    cloud_pb = o_pb2.PointCloud(header=header,
                                points=points_pb)
    return cloud_pb

### Vision Messages ###
def detection3d_to_proto(d3d_msg, class_names,
                         target_frame=None, tf2buf=None):
    """Given vision_msgs.Detection3D, return a proto Detection object with a 3D
    box. because the label in d3d_msg have integer id, we will need to map them
    to strings according to indexing in 'class_names'.

    If the message contains multiple object hypotheses, will only
    consider the one with the highest score
    """
    hypos = {h.hypothesis.class_id: h.hypothesis.score
             for h in d3d_msg.results}
    label_id = max(hypos, key=hypos.get)
    if type(label_id) == int:
        label = class_names[label_id]
    elif type(label_id) == str:
        label = label_id
    confidence = hypos[label_id]
    bbox_center = d3d_msg.bbox.center  # a Pose msg

    # transform pose to target frame if wanted
    if target_frame is not None:
        if tf2buf is None:
            tf2buf = tf2_ros.Buffer()
        bbox_center_stamped = geometry_msgs.msg.PoseStamped(header=d3d_msg.header, pose=bbox_center)
        bbox_center_stamped_T_target = tf2_transform(tf2buf, bbox_center_stamped, target_frame)
        bbox_center = bbox_center_stamped_T_target.pose
    center_tuple = pose_to_tuple(bbox_center)
    center_pb = proto_utils.posetuple_to_poseproto(center_tuple)
    box = common_pb2.Box3D(center=center_pb,
                           sizes=common_pb2.Vec3(x=d3d_msg.bbox.size.x,
                                                 y=d3d_msg.bbox.size.y,
                                                 z=d3d_msg.bbox.size.z))
    return o_pb2.Detection(label=label,
                           confidence=confidence,
                           box_3d=box)

def detection3darray_to_proto(d3darr_msg, robot_id, class_names,
                              target_frame=None, tf2buf=None):
    """Given a vision_msgs.Detection3DArray message,
    return an ObjectDetectionArray proto. 'robot_id'
    is the robot that made this detetcion"""
    stamp = google.protobuf.timestamp_pb2.Timestamp(seconds=d3darr_msg.header.stamp.sec,
                                                    nanos=d3darr_msg.header.stamp.nanosec)
    if target_frame is None:
        header = proto_utils.make_header(frame_id=d3darr_msg.header.frame_id, stamp=stamp)
    else:
        header = proto_utils.make_header(frame_id=target_frame, stamp=stamp)
    detections_pb = []
    for d3d_msg in d3darr_msg.detections:
        det3d_pb = detection3d_to_proto(
            d3d_msg, class_names, target_frame=target_frame, tf2buf=tf2buf)
        detections_pb.append(det3d_pb)
    return o_pb2.ObjectDetectionArray(header=header,
                                      robot_id=robot_id,
                                      detections=detections_pb)


### MISC ###
def unravel_args(fields, tup):
    """convenient function that outputs a 1-1 dictionary
    between the given fields and tuple"""
    return {fields[i]:tup[i] for i in range(len(tup))}
