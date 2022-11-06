# ROS Utilities
import sys
import math
import numpy as np
import pickle

import rospy
import geometry_msgs
import std_msgs
import visualization_msgs
import message_filters
import vision_msgs

import tf
import tf2_ros
#!!! NEED THIS:
# https://answers.ros.org/question/95791/tf-transformpoint-equivalent-on-tf2/?answer=394789#post-id-394789
# STUPID ROS PROBLEM.
import tf2_geometry_msgs.tf2_geometry_msgs

# we don't want to crash if a ros-related package is not installed.
import importlib
if importlib.util.find_spec("ros_numpy") is not None:
    import ros_numpy

import google.protobuf.timestamp_pb2
import google.protobuf.any_pb2
import sloop_object_search.grpc.common_pb2 as common_pb2
import sloop_object_search.grpc.observation_pb2 as o_pb2
import sloop_object_search.grpc.common_pb2 as c_pb2
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.colors import color_map, cmaps, lighter_with_alpha


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

def tf2_do_transform(pose_stamped, trans):
    return tf2_geometry_msgs.do_transform_pose(pose_stamped, trans)

### Mathematics ###
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
    return 2*math.atan2(math_utils.vec_norm(qd[:3]), qd[3])

def quat_diff_angle_relative(q1, q2):
    """returns the angle (radians) between q1 and q2;
    The angle will be the smallest one between the two
    vectors. It is the "intuitive" angular difference.
    Unsigned."""
    # reference: https://stackoverflow.com/a/23263233/2893053
    ad = quat_diff_angle(q1, q2)
    return min(2*math.pi - ad, ad)


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

def pointcloud2_to_pointcloudproto(cloud_msg):
    """
    Converts a PointCloud2 message to a PointCloud proto message.

    Args:
       cloud_msg (sensor_msgs.PointCloud2)
    """
    pcl_raw_array = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
    points_xyz_array = ros_numpy.point_cloud2.get_xyz_points(pcl_raw_array)

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
    hypos = {h.id: h.score for h in d3d_msg.results}
    label_id = max(hypos, key=hypos.get)
    label = class_names[label_id]
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
    stamp = google.protobuf.timestamp_pb2.Timestamp(seconds=d3darr_msg.header.stamp.secs,
                                                    nanos=d3darr_msg.header.stamp.nsecs)
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

def tf2msg_from_object_loc(loc, world_frame, object_frame, **kwargs):
    stamp = kwargs.get("stamp", rospy.Time.now())
    t = geometry_msgs.msg.TransformStamped(
        header=std_msgs.msg.Header(stamp=stamp,
                                   frame_id=world_frame))
    t.child_frame_id = object_frame
    t.transform.translation = geometry_msgs.msg.Vector3(x=loc[0], y=loc[1], z=loc[2])
    t.transform.rotation = geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
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

def make_viz_marker_for_line_segment(start_point, end_point, header, **kwargs):
    marker = Marker(header=header)
    _id = kwargs.pop("id", hash16((start_point, end_point)))
    marker.id = _id
    marker.points = [geometry_msgs.msg.Point(*start_point),
                     geometry_msgs.msg.Point(*end_point)]
    marker.pose.position = geometry_msgs.msg.Point(0, 0, 0)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    kwargs["viz_type"] = Marker.LINE_STRIP
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_cylinder(_id, pose, header, **kwargs):
    """generic method for cylinder; note that pose should be a 7-d tuple"""
    marker = Marker(header=header)
    marker.id = _id
    x, y, z, qx ,qy ,qz, qw = pose
    marker.pose.position = geometry_msgs.msg.Point(x,y,z)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
    kwargs["viz_type"] = Marker.CYLINDER
    _fill_viz_marker(marker, **kwargs)
    return marker

def make_viz_marker_cube(_id, pose, header, **kwargs):
    """generic method for cylinder; note that pose should be a 7-d tuple"""
    marker = Marker(header=header)
    marker.id = _id
    x, y, z, qx ,qy ,qz, qw = pose
    marker.pose.position = geometry_msgs.msg.Point(x,y,z)
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=qx, y=qy, z=qz, w=qw)
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
    marker.lifetime = rospy.Duration(lifetime)
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
    marker.pose.orientation = geometry_msgs.msg.Quaternion(x=0, y=0, z=0, w=1)
    marker.scale = geometry_msgs.msg.Vector3(x=res, y=res, z=res)
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(lifetime)
    marker.color = std_msgs.msg.ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
    return marker

def clear_markers(header, ns):
    """BUG: RVIZ actually clears all messages"""
    marker = Marker()
    marker.header = header
    marker.ns = ns
    marker.action = Marker.DELETEALL
    return MarkerArray([marker])

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
    return MarkerArray(markers)

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
    return MarkerArray(markers)

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
    return MarkerArray(markers)


def viz_msgs_for_robot_pose(robot_pose, world_frame, robot_frame, stamp=None,
                            **kwargs):
    """Given a robot pose in the world frame obtained from SLOOP,
    return a tuple (marker, tf2msg) for visualization. Note that
    this function accounts for the differences in default look
    direction between ROS and SLOOP."""
    if stamp is None:
        stamp = rospy.Time.now()
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
    tf2msg = tf2msg_from_robot_pose(robot_pose, world_frame, robot_frame)
    return marker, tf2msg


def viz_msgs_for_robot_pose_proto(robot_pose_proto, world_frame, robot_frame, stamp=None,
                                  **kwargs):
    """Given a robot_pose_proto obtained from sloop grpc server,
    return a tuple (marker, tf2msg) for visualization. Note that
    this function accounts for the differences in default look
    direction between ROS and SLOOP."""
    robot_pose = proto_utils.robot_pose_from_proto(robot_pose_proto)
    return viz_msgs_for_robot_pose(robot_pose, world_frame, robot_frame, stamp=stamp, **kwargs)


### Communication ###
class WaitForMessages:
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
        _start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.messages is not None:
                rospy.loginfo("WaitForMessages: Received messages! Done!")
                break
            if self.verbose:
                rospy.loginfo("WaitForMessages: waiting for messages from {}".format(topics))
            _dt = rospy.Time.now() - _start
            if timeout is not None and _dt.to_sec() > timeout:
                rospy.logerr("WaitForMessages: timeout waiting for messages")
                self.messages = [None]*len(topics)
                if exception_on_timeout:
                    raise TimeoutError("WaitForMessages: timeout waiting for messages")
                break
            rate.sleep()

    def _cb(self, *messages):
        if self.messages is not None:
            return
        if self.verbose:
            rospy.loginfo("WaitForMessages: got messages!")
        self.messages = messages


### Time ###
def stamp_to_sec(stamp):
    return stamp.to_sec()
