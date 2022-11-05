# To run test (for UR5):
# 1. run in a terminal 'python -m sloop_object_search.grpc.server'
# 2. run in a terminal 'python sloop_mos_viam.py'

import asyncio

import math
import yaml
import numpy as np
from pypcd import pypcd

import open3d as o3d
import base64

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionServiceClient, VisModelConfig, VisModelType
from viam.services.motion import MotionServiceClient

from viam.proto.common import ResourceName, PoseInFrame

from viam_utils import OrientationVector

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

########### visualization ###########
def get_and_visualize_belief():
    raise NotImplementedError




########### procedural methods ###########
def server_message_callback(message):
    print(message)
    raise NotImplementedError()

def update_search_region(robot_id, agent_config, sloop_client):
    print("Sending request to update search region (3D)")

    cloud_arr = viam_get_point_cloud_array()
    cloud_pb = proto_utils.pointcloudproto_from_array(cloud_arr)

    robot_pose = viam_get_ee_pose()
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


########### main object search logic ###########
async def run_sloop_search(viam_robot,
                           config,
                           world_frame=None,
                           dynamic_update=False):
    """config: configuration dictionary for SLOOP"""
    sloop_client = SloopObjectSearchClient()
    agent_config = config["agent_config"]
    planner_config = config["planner_config"]
    robot_id = agent_config["robot"]["id"]

    last_action = None
    objects_found = set()
    #-----------------------------------------

    # await viam_get_object_detections2d(viam_robot)
    # await viam_get_ee_pose(viam_robot)
    # await viam_get_point_cloud_array(viam_robot)

    await viam_move_ee_to(viam_robot, None, None, None)


    # # First, create an agent
    # sloop_client.createAgent(
    #     header=proto_utils.make_header(), config=agent_config,
    #     robot_id=robot_id)

    # # Make the client listen to server
    # ls_future = sloop_client.listenToServer(
    #     robot_id, server_message_callback)
    # local_robot_id = None  # needed if the planner is hierarchical

    # # Update search region
    # update_search_region(robot_id, agent_config, sloop_client)

    # # wait for agent creation
    # print("waiting for sloop agent creation...")
    # sloop_client.waitForAgentCreation(robot_id)
    # print("agent created!")

    # # visualize initial belief
    # get_and_visualize_belief()

    # # create planner
    # response = sloop_client.createPlanner(config=planner_config,
    #                                       header=proto_utils.make_header(),
    #                                       robot_id=robot_id)
    # rospy.loginfo("planner created!")

    # # Send planning requests
    # for step in range(config["task_config"]["max_steps"]):
    #     action_id, action_pb = plan_action()
    #     execute_action(action_id, action_pb)

    #     if dynamic_update:
    #         update_search_region(robot_id, agent_config, sloop_client)

    #     response_observation, response_robot_belief =\
    #         self.wait_observation_and_update_belief(action_id)
    #     print(f"Step {step} robot belief:")
    #     robot_belief_pb = response_robot_belief.robot_belief
    #     objects_found = set(robot_belief_pb.objects_found.object_ids)
    #     objects_found.update(objects_found)
    #     print(f"  pose: {robot_belief_pb.pose.pose_3d}")
    #     print(f"  objects found: {objects_found}")
    #     print("-----------")

    #     # visualize FOV and belief
    #     self.get_and_visualize_belief()
    #     if response_observation.HasField("fovs"):
    #         self.visualize_fovs_3d(response_observation)

    #     # Check if we are done
    #     if objects_found == set(self.agent_config["targets"]):
    #         rospy.loginfo("Done!")
    #         break
    #     time.sleep(1)

async def test_ur5e_viamlab():
    with open("./config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)

    print(">>>>>>><<<<<<<<>>>> viam connecting >><<<<<<<<>>>>>>>")
    viam_robot = await viam_connect()
    print('Resources:')
    print(viam_robot.resource_names)

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    await run_sloop_search(viam_robot,
                           config,
                           world_frame="base",
                           dynamic_update=False)

if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
