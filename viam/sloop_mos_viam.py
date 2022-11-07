# To run test (for UR5):
# 1. run in a terminal 'python -m sloop_object_search.grpc.server'
# 2. run in a terminal 'python sloop_mos_viam.py'

import asyncio

import math
import yaml
import numpy as np

import open3d as o3d
import base64

from sloop_object_search.grpc.client import SloopObjectSearchClient
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc import sloop_object_search_pb2 as slpb2
from sloop_object_search.grpc import observation_pb2 as o_pb2
from sloop_object_search.grpc import action_pb2 as a_pb2
from sloop_object_search.grpc import common_pb2
from sloop_object_search.grpc.common_pb2 import Status
from sloop_object_search.grpc.constants import Message
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils.misc import import_class

import viam_utils

########### visualization ###########
def get_and_visualize_belief():
    raise NotImplementedError




########### procedural methods ###########
def server_message_callback(message):
    print(message)
    raise NotImplementedError()

def update_search_region(viam_robot, agent_config, sloop_client):
    print("Sending request to update search region (3D)")
    robot_id = agent_config["robot"]["id"]

    if viam_utils.MOCK:
        cloud_arr = np.array([])
        robot_pose = (0.2797589770640316, 0.7128048233719448, 0.5942370817926967,
                      -0.6500191634979094, 0.4769735333791088,
                      0.4926158987104014, 0.32756817897816304)
    else:
        try:
            cloud_arr = viam_get_point_cloud_array()
        except AssertionError:
            print("Failed to obtain point cloud. Will proceed with empty point cloud.")
            cloud_arr = np.array([])
        robot_pose = viam_get_ee_pose(viam_robot)

    cloud_pb = proto_utils.pointcloudproto_from_array(cloud_arr)
    robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

    # parameters
    search_region_config = agent_config.get("search_region", {}).get("3d", {})
    search_region_params_3d = dict(
        octree_size=search_region_config.get("octree_size", 32),
        search_space_resolution=search_region_config.get("res", SEARCH_SPACE_RESOLUTION_3D),
        region_size_x=search_region_config.get("region_size_x"),
        region_size_y=search_region_config.get("region_size_y"),
        region_size_z=search_region_config.get("region_size_z"),
        debug=search_region_config.get("debug", False)
    )
    sloop_client.updateSearchRegion(
        header=cloud_pb.header,
        robot_id=robot_id,
        robot_pose=robot_pose_pb,
        point_cloud=cloud_pb,
        search_region_params_3d=search_region_params_3d)

def plan_action(robot_id, world_frame, sloop_client):
    response_plan = sloop_client.planAction(
        robot_id, header=proto_utils.make_header(world_frame))
    action_pb = proto_utils.interpret_planned_action(response_plan)
    action_id = response_plan.action_id
    rospy.loginfo("plan action finished. Action ID: {}".format(typ.info(action_id)))
    self.last_action = action_pb
    return action_id, action_pb


def execute_action(action_id, action_pb):
    """All viewpoint movement actions specify a goal pose
    the robot should move its end-effector to, and publish
    that as a KeyValAction."""
    if isinstance(action_pb, a_pb2.MoveViewpoint):
        if action_pb.HasField("dest_3d"):
            dest = proto_utils.poseproto_to_posetuple(action_pb.dest_3d)
            nav_type = "3d"
        elif action_pb.HasField("dest_2d"):
            raise NotImplementedError("Not expecting destination to be 2D")
        else:
            raise NotImplementedError("Not implemented action_pb.")

        # TODO: action execution
        print("Executing nav action (viewpoint movement)")
        viam_move_ee_to(dest[:3], dest[3:], action_id)

    elif isinstance(action_pb, a_pb2.Find):
        print("Signaling find action")
        viam_signal_find(action_id)

    # TODO: wait for action is done (this may be part of viam_xxx functions)

def wait_for_observation(robot_id, last_action,
                         agent_config, world_frame, objects_found):
    """We wait for the robot pose (PoseStamped) and the
    object detections (vision_msgs.Detection3DArray)

    Returns:
        a tuple: (detections_pb, robot_pose_pb, objects_found_pb)"""
    # TODO: future viam: time sync between robot pose and object detection
    robot_pose = viam_get_ee_pose()
    robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
    detections = viam_get_object_detections()

    # Detection proto
    detections_pb = detections_to_proto(robot_id, detections)

    # Objects found proto
    # If the last action is "find", and we receive object detections
    # that contain target objects, then these objects will be considered 'found'
    if isinstance(last_action, a_pb2.Find):
        for det_pb in detections_pb.detections:
            if det_pb.label in agent_config["targets"]:
                objects_found.add(det_pb.label)
    header = proto_utils.make_header(frame_id=world_frame)
    objects_found_pb = o_pb2.ObjectsFound(
        header=header, robot_id=robot_id,
        object_ids=sorted(list(objects_found)))

    # Robot pose proto
    robot_pose_tuple = ros_utils.pose_to_tuple(robot_pose_msg.pose)
    robot_pose_pb = o_pb2.RobotPose(
        header=header,
        robot_id=robot_id,
        pose_3d=proto_utils.posetuple_to_poseproto(robot_pose_tuple))
    return detections_pb, robot_pose_pb, objects_found_pb

def wait_observation_and_update_belief(robot_id, world_frame, action_id, sloop_client):
    # Now, wait for observation, and then update belief
    detections_pb, robot_pose_pb, objects_found_pb = wait_for_observation()
    # send obseravtions for belief update
    header = proto_utils.make_header(frame_id=world_frame)
    response_observation = sloop_client.processObservation(
        robot_id, robot_pose_pb,
        object_detections=detections_pb,
        objects_found=objects_found_pb,
        header=header, return_fov=True,
        action_id=action_id, action_finished=True, debug=False)
    response_robot_belief = sloop_client.getRobotBelief(
        robot_id, header=proto_utils.make_header(world_frame))
    return response_observation, response_robot_belief




if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
