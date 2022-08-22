import numpy as np

# we don't want to crash if a ros-related package is not installed.
import importlib
if importlib.util.find_spec("ros_numpy") is not None:
    import ros_numpy

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.any_pb2 import Any
import logging

import pomdp_py

import sloop_object_search.grpc.observation_pb2 as o_pb2
from sloop_object_search.grpc.common_pb2\
    import Vec2, Vec3, Header, Pose2D, Pose3D, Quaternion, Histogram, Voxel3D
from sloop_object_search.grpc.action_pb2\
    import MoveViewpoint, Find, KeyValueAction, Motion2D, Motion3D
from .. import sloop_object_search_pb2 as slpb2

from sloop_object_search.oopomdp.domain import action as sloop_action
from sloop_object_search.oopomdp.domain import observation as sloop_observation
from sloop_object_search.oopomdp.models.search_region import SearchRegion3D
from sloop_object_search.oopomdp.models.octree_belief import Octree, OctreeBelief
from sloop_object_search.utils.math import to_rad, fround
from sloop_object_search.utils import open3d_utils

def v3toa(v3):
    """convert Vec3 proto to numpy array"""
    return np.array([v3.x, v3.y, v3.z])

def process_search_region_params_2d(search_region_params_2d_pb):
    params = {}
    if search_region_params_2d_pb.HasField('layout_cut'):
        params["layout_cut"] = search_region_params_2d_pb.layout_cut
    if search_region_params_2d_pb.HasField('floor_cut'):
        params["floor_cut"] = search_region_params_2d_pb.floor_cut
    if search_region_params_2d_pb.HasField('grid_size'):
        params["grid_size"] = search_region_params_2d_pb.grid_size
    if search_region_params_2d_pb.HasField('brush_size'):
        params["brush_size"] = search_region_params_2d_pb.brush_size
    if search_region_params_2d_pb.HasField('region_size'):
        params["region_size"] = search_region_params_2d_pb.region_size
    if search_region_params_2d_pb.HasField('debug'):
        params["debug"] = search_region_params_2d_pb.debug
    return params

def process_search_region_params_3d(search_region_params_3d_pb):
    params = {}
    if search_region_params_3d_pb.HasField('octree_size'):
        params["octree_size"] = search_region_params_3d_pb.octree_size
    if search_region_params_3d_pb.HasField('search_space_resolution'):
        params["search_space_resolution"] = search_region_params_3d_pb.search_space_resolution
    if search_region_params_3d_pb.HasField('debug'):
        params["debug"] = search_region_params_3d_pb.debug
    if search_region_params_3d_pb.HasField('region_size_x'):
        params["region_size_x"] = search_region_params_3d_pb.region_size_x
    if search_region_params_3d_pb.HasField('region_size_y'):
        params["region_size_y"] = search_region_params_3d_pb.region_size_y
    if search_region_params_3d_pb.HasField('region_size_z'):
        params["region_size_z"] = search_region_params_3d_pb.region_size_z
    return params


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
        point_pb = o_pb2.PointCloud.Point(pos=Vec3(x=p[0], y=p[1], z=p[2]))
        points_pb.append(point_pb)

    header = Header(stamp=Timestamp().GetCurrentTime(),
                    frame_id=cloud_msg.header.frame_id)
    cloud_pb = o_pb2.PointCloud(header=header,
                                points=points_pb)
    return cloud_pb


def pointcloudproto_to_array(point_cloud):
    points_array = np.array([[p.pos.x, p.pos.y, p.pos.z]
                             for p in point_cloud.points])
    return points_array

def posemsg_to_pose3dproto(pose_msg):
    return Pose3D(position=Vec3(x=pose_msg.position.x,
                                y=pose_msg.position.y,
                                z=pose_msg.position.z),
                  rotation=Quaternion(x=pose_msg.orientation.x,
                                      y=pose_msg.orientation.y,
                                      z=pose_msg.orientation.z,
                                      w=pose_msg.orientation.w))

def posetuple_to_poseproto(pose):
    if len(pose) == 7:
        x,y,z,qx,qy,qz,qw = pose
        return Pose3D(position=Vec3(x=x, y=y, z=z),
                      rotation=Quaternion(x=qx, y=qy, z=qz, w=qw))
    elif len(pose) == 3:
        x,y,th = pose
        return Pose2D(x=x, y=y, th=th)
    else:
        raise ValueError(f"Invalid pose: {pose}")

def quatproto_to_tuple(quat):
    return (quat.x, quat.y, quat.z, quat.w)

def make_header(frame_id=None, stamp=None):
    if stamp is None:
        stamp = Timestamp().GetCurrentTime()
    if frame_id is None:
        return Header(stamp=stamp)
    else:
        return Header(stamp=stamp, frame_id=frame_id)

def robot_pose_from_proto(robot_pose_pb):
    """returns a tuple representation from RobotPose proto.

    If it is 2D, then return (x, y, th). If it is 3D, then return (x, y, z, qx,
    qy, qz, qw).
    """
    if not isinstance(robot_pose_pb, o_pb2.RobotPose):
        raise TypeError("robot_pose_pb should be RobotPose")
    if robot_pose_pb.HasField("pose_2d"):
        pose2d = robot_pose_pb.pose_2d
        return (pose2d.x, pose2d.y, pose2d.th)
    elif robot_pose_pb.HasField("pose_3d"):
        pose3d = robot_pose_pb.pose_3d
        return (pose3d.position.x, pose3d.position.y, pose3d.position.z,
                pose3d.rotation.x, pose3d.rotation.y, pose3d.rotation.z,
                pose3d.rotation.w)
    elif robot_pose_pb.HasField("pose_particles"):
        raise NotImplementedError("can't handle 'pose_particles' yet")
    elif robot_pose_pb.HasField("pose_hist"):
        raise NotImplementedError("can't handle 'pose_hist' yet")
    else:
        raise ValueError("request does not contain valid robot pose field.")
    return None

def robot_pose_proto_from_tuple(robot_pose):
    """Returns a RobotPose proto from a given tuple
    representation of robot pose."""
    if len(robot_pose) == 3:
        return o_pb2.RobotPose(pose_2d=posetuple_to_poseproto(robot_pose))
    elif len(robot_pose) == 7:
        return o_pb2.RobotPose(pose_3d=posetuple_to_poseproto(robot_pose))
    else:
        raise ValueError(f"Invalid pose: {robot_pose}")

def pomdp_action_to_proto(action, agent, header=None):
    if header is None:
        header = make_header()
    if isinstance(action, sloop_action.MotionAction):
        action_type = "move_action"
        if isinstance(action, sloop_action.MotionAction2D):
            raise NotImplementedError()
        elif isinstance(action, sloop_action.MotionActionTopo):
            raise NotImplementedError()
        elif isinstance(action, sloop_action.MotionAction3D):
            dpos_pomdp, drot = action.motion
            # we need to convert the position change from pomdp frame to
            # the world frame.
            dpos_world = agent.search_region.to_world_pos(dpos_pomdp)
            motion_pb = Motion3D(
                dpos=Vec3(x=dpos_world[0], y=dpos_world[1], z=dpos_world[2]),
                drot_euler=Vec3(x=to_rad(drot[0]), y=to_rad(drot[1]), z=to_rad(drot[2])))
            action_pb = MoveViewpoint(header=header,
                                      robot_id=agent.robot_id,
                                      motion_3d=motion_pb,
                                      name=action.name,
                                      expected_cost=action.step_cost)
    elif isinstance(action, sloop_action.FindAction):
        action_type = "find_action"
        action_pb = Find(header=header,
                         robot_id=agent.robot_id,
                         name=action.name)
    else:
        raise RuntimeError(f"Unrecognized action {action}")
    return action_type, action_pb


def interpret_planned_action(plan_action_reply):
    """Given the response from PlanActionReply,
    return the protobuf object corresponding to
    the action."""
    assert isinstance(plan_action_reply, slpb2.PlanActionReply),\
        "only interprets PlanActionReply"
    if plan_action_reply.HasField("move_action"):
        return plan_action_reply.move_action
    elif plan_action_reply.HasField("find_action"):
        return plan_action_reply.find_action
    elif plan_action_reply.HasField("kv_action"):
        return plan_action_reply.kv_action
    else:
        raise ValueError("unable to determine action.")


def pomdp_object_beliefs_to_proto(object_beliefs, search_region):
    """
    Args:
        object_beliefs; Maps from objid to a pomdp_py.GenerativeDistribution
    Return:
        A list of ObjectBelief protos
    """
    object_beliefs_proto = []
    for objid in object_beliefs:
        b_obj = object_beliefs[objid]

        hist_values = []  # the search region locations
        hist_probs = []   # the probabilities
        if not isinstance(search_region, SearchRegion3D):
            # For 2D belief, just iterate over all
            for s_obj in b_obj:
                assert s_obj.is_2d, "expecting object state to be 2d."
                x, y = search_region.to_world_pos(s_obj.loc)
                hist_values.append(Vec2(x=x, y=y))
                hist_probs.append(b_obj[s_obj])

        else:
            # b_obj is octree belief
            assert isinstance(b_obj, OctreeBelief),\
                "3d object belief should be octree belief"

            # each voxel is (x,y,z,r,_) where x,y,z are ground-level voxel coordinates.
            open3d_utils.draw_octree_dist(b_obj.octree_dist)
            voxels = b_obj.octree_dist.collect_plotting_voxels()
            probs = [b_obj.octree_dist.prob_at(*Octree.increase_res(voxels[i][:3], 1, voxels[i][3]), voxels[i][3])
                     for i in range(len(voxels))]
            for i in range(len(voxels)):
                vpos = voxels[i][:3]  # voxel location at ground-level (but in pomdp frame)
                vres = voxels[i][3]
                x, y, z = search_region.to_world_pos(vpos)
                res = vres * search_region.search_space_resolution
                voxel_pb = Voxel3D(pos=Vec3(x=x, y=y, z=z), res=res)
                hist_values.append(to_any_proto(voxel_pb))
                hist_probs.append(probs[i])

        dist = Histogram(length=len(hist_values),
                         values=hist_values,
                         probs=hist_probs)
        object_beliefs_proto.append(
            slpb2.ObjectBelief(object_id=objid,
                               dist=dist))
    return object_beliefs_proto


def to_any_proto(val):
    val_any = Any()
    val_any.Pack(val)
    return val_any


def robot_belief_to_proto(robot_belief, header=None):
    """Given a pomdp_py.WeightedParticles or pomdp_py.Histogram
    representation of robot belief, return a RobotBelief proto.
    Uncertainty over the robot belief is possibly in its pose. We
    assume the robot observes its other attributes such as 'objects_found'."""
    if not isinstance(robot_belief, pomdp_py.WeightedParticles)\
       and not isinstance(robot_belief, pomdp_py.Histogram):
        raise TypeError("robot_belief should be either "
                        "pomdp_py.WeightedParticles or pomdp_py.Histogram")
    if header is None:
        header = make_header()
    # For now, we return the most likely robot state, although
    # the protobuf definition is more general.
    mpe_robot_state = robot_belief.mpe()
    robot_id = mpe_robot_state["id"]
    if mpe_robot_state.is_2d:
        pose_field = {"pose_2d": posetuple_to_poseproto(mpe_robot_state.pose)}
    else:
        pose_field = {"pose_3d": posetuple_to_poseproto(mpe_robot_state.pose)}
    robot_pose_pb = o_pb2.RobotPose(header=header,
                                    robot_id=robot_id,
                                    **pose_field)
    return slpb2.RobotBelief(robot_id=robot_id,
                             objects_found=list(map(str, mpe_robot_state.objects_found)),
                             pose=robot_pose_pb)

def pomdp_detection_from_proto(detection_pb, search_region,
                               pos_precision='int',
                               rot_precision=0.001,
                               size_precision=0.001):
    """given Detection3D proto, return ObjectDetection object
    The pose in the detection will be converted to POMDP space.
    Its position and orientation will be rounded to the specified
    precision."""
    objid = detection_pb.label
    center = detection_pb.box.center
    sizes = v3toa(detection_pb.box.sizes)

    # because the POMDP frame and the world frame are axis-aligned,
    # we only need to convert the position, not rotation.
    center_pos = (center.position.x, center.position.y, center.position.z)
    center_rot = fround(rot_precision, quatproto_to_tuple(center.rotation))
    pomdp_center_pos = fround(pos_precision, search_region.to_pomdp_pos(center_pos))
    pomdp_sizes = fround(size_precision, sizes / search_region.search_space_resolution)
    pomdp_pose = (pomdp_center_pos, center_rot)
    return sloop_observation.ObjectDetection(objid, pomdp_pose, sizes=pomdp_sizes)


def pomdp_observation_from_proto(observation_pb, search_region, **kwargs):
    if isinstance(observation_pb, o_pb2.ObjectDetectionArray):
        objobzs = {}
        for detection_pb in observation_pb.detections:
            objo = pomdp_detection_from_proto(
                detection_pb, search_region, **kwargs)
            if objo.id not in objobzs:
                objobzs[objo.id] = objo
            else:
                logging.warning(f"multiple detections for {objo.id}. Only keeping one.")
        return sloop_observation.JointObservation(objobzs)

    elif isinstance(observation_pb, o_pb2.RobotPose):
        raise NotImplementedError
    elif isinstance(observation_pb, o_pb2.Language):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported observation type {type(observation_pb)}")
