import numpy as np

# we don't want to crash if a ros-related package is not installed.
import importlib
if importlib.util.find_spec("ros_numpy") is not None:
    import ros_numpy

from google.protobuf.timestamp_pb2 import Timestamp

from sloop_object_search.grpc.observation_pb2 import PointCloud
from sloop_object_search.grpc.common_pb2\
    import Vec3, Header, Pose3D, Quaternion
from sloop_object_search.grpc.action_pb2\
    import MoveViewpoint, Find, KeyValueAction, Motion2D, Motion3D
from .. import sloop_object_search_pb2 as slpb2

from sloop_object_search.oopomdp.domain import action as sloop_action
from sloop_object_search.utils.math import to_rad


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
        point_pb = PointCloud.Point(pos=Vec3(x=p[0], y=p[1], z=p[2]))
        points_pb.append(point_pb)

    header = Header(stamp=Timestamp().GetCurrentTime(),
                    frame_id=cloud_msg.header.frame_id)
    cloud_pb = PointCloud(header=header,
                          points=points_pb)
    return cloud_pb


def pointcloudproto_to_array(point_cloud):
    points_array = np.array([[p.pos.x, p.pos.y, p.pos.z]
                             for p in point_cloud.points])
    return points_array

def pose_to_pose3dproto(pose_msg):
    return Pose3D(position=Vec3(x=pose_msg.position.x,
                                y=pose_msg.position.y,
                                z=pose_msg.position.z),
                  rotation=Quaternion(x=pose_msg.orientation.x,
                                      y=pose_msg.orientation.y,
                                      z=pose_msg.orientation.z,
                                      w=pose_msg.orientation.w))


def make_header(frame_id=None, stamp=None):
    if stamp is None:
        stamp = Timestamp().GetCurrentTime()
    if frame_id is None:
        return Header(stamp=stamp)
    else:
        return Header(stamp=stamp, frame_id=frame_id)

def interpret_robot_pose(request):
    """Given a request proto object whose definition contains

        oneof robot_pose {
            Pose2D robot_pose_2d = 4;
            Pose3D robot_pose_3d = 5;
          }

    return a tuple representation of the pose. If it is 2D,
    then return (x, y, th). If it is 3D, then return (x, y, z,
    qx, qy, qz, qw).
    """
    if request.HasField('robot_pose_2d'):
        robot_pose = (request.robot_pose_2d.x,
                      request.robot_pose_2d.y,
                      request.robot_pose_2d.th)
    elif request.HasField('robot_pose_3d'):
        robot_pose = (request.robot_pose_3d.position.x,
                      request.robot_pose_3d.position.y,
                      request.robot_pose_3d.position.z,
                      request.robot_pose_3d.rotation.x,
                      request.robot_pose_3d.rotation.y,
                      request.robot_pose_3d.rotation.z,
                      request.robot_pose_3d.rotation.w,)
    else:
        raise ValueError("request does not contain valid robot pose field.")
    return robot_pose


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
