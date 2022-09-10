#!/usr/bin/env python
# Test the complete algorithm: hierarchical search.
# Performs hierarchical object search with 3D local
# search and 2D global search.
#
#
# To run the test, do the following IN ORDER:
# 0. run config_simple_sim_lab121_lidar.py to generate the .yaml configuration file
# 1. run in a terminal 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=lab121_lidar'
# 2. run in a terminal 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose'
# 3. run in a terminal 'roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=lab121_lidar'
# 4. run in a terminal 'python -m sloop_object_search.grpc.server'
# 5. run in a terminal 'python test_simple_sim_env_hier_search.py'
# 6. run in a terminal 'roslaunch sloop_object_search_ros view_simple_sim.launch'
# ------------------
import numpy as np
import time
import rospy
import pickle
import json
import asyncio
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
from visualization_msgs.msg import Marker, MarkerArray
from sloop_object_search_ros.msg import KeyValAction, KeyValObservation
from sloop_object_search.grpc.client import SloopObjectSearchClient
from sloop_object_search.grpc.utils import proto_utils
from sloop_mos_ros import ros_utils
from sloop_object_search.utils.open3d_utils import draw_octree_dist
from sloop_object_search.grpc import sloop_object_search_pb2 as slpb2
from sloop_object_search.grpc import observation_pb2 as o_pb2
from sloop_object_search.grpc import action_pb2 as a_pb2
from sloop_object_search.grpc import common_pb2
from sloop_object_search.grpc.constants import Message
from sloop_object_search.grpc.common_pb2 import Status
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils
from test_simple_sim_env_navigation import make_nav_action
from test_simple_sim_env_common import (TestSimpleEnvCase,
                                        observation_msg_to_proto,
                                        wait_for_robot_pose,
                                        REGION_POINT_CLOUD_TOPIC,
                                        ROBOT_POSE_TOPIC,
                                        ACTION_DONE_TOPIC, OBSERVATION_TOPIC,
                                        AGENT_CONFIG, PLANNER_CONFIG,
                                        TASK_CONFIG)


class TestSimpleEnvHierSearch(TestSimpleEnvCase):

    def server_message_callback(self, message):
        if Message.match(message) == Message.REQUEST_LOCAL_SEARCH_REGION_UPDATE:
            local_robot_id = Message.forwhom(message)
            rospy.loginfo(f"will send a update search request to {local_robot_id}")
            self.update_search_region_3d(robot_id=local_robot_id)

    def __init__(self, prior="uniform"):
        super().__init__(prior=prior)

        # Make the client listen to server
        ls_future = self._sloop_client.listenToServer(
            self.robot_id, self.server_message_callback)

        self.update_search_region_2d()

        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief_2d()

        # create planner
        response = self._sloop_client.createPlanner(config=PLANNER_CONFIG,
                                                    header=proto_utils.make_header(),
                                                    robot_id=self.robot_id)
        rospy.loginfo("planner created!")

        # Send planning requests
        for step in range(TASK_CONFIG["max_steps"]):

            response_plan = self._sloop_client.planAction(
                self.robot_id, header=proto_utils.make_header(self.world_frame))
            action = proto_utils.interpret_planned_action(response_plan)
            action_id = response_plan.action_id
            rospy.loginfo("plan action finished. Action ID: {}".format(action_id))

            # Now, we need to execute the action, and receive observation
            # from SimpleEnv. First, convert the dest_3d in action to
            # a KeyValAction message
            if isinstance(action, a_pb2.MoveViewpoint):
                if action.HasField("dest_3d"):
                    dest = proto_utils.poseproto_to_posetuple(action.dest_3d)

                elif action.HasField("dest_2d"):
                    robot_pose = np.asarray(wait_for_robot_pose())
                    dest_2d = proto_utils.poseproto_to_posetuple(action.dest_2d)
                    x, y, thz = dest_2d
                    z = robot_pose[2]
                    thx, thy, _ = math_utils.quat_to_euler(*robot_pose[3:])
                    dest = (x, y, z, *math_utils.euler_to_quat(thx, thy, thz))

                elif action.HasField("motion_3d"):
                    robot_pose = np.asarray(wait_for_robot_pose())
                    robot_pose[0] += action.motion_3d.dpos.x
                    robot_pose[1] += action.motion_3d.dpos.y
                    robot_pose[2] += action.motion_3d.dpos.z

                    thx, thy, thz = math_utils.quat_to_euler(*robot_pose[3:])
                    dthx = action.motion_3d.drot_euler.x
                    dthy = action.motion_3d.drot_euler.y
                    dthz = action.motion_3d.drot_euler.z
                    robot_pose[3:] = np.asarray(math_utils.euler_to_quat(
                        thx + dthx, thy + dthy, thz + dthz))
                    dest = robot_pose

                elif action.HasField("motion_2d"):
                    forward = action.motion_2d.forward
                    angle = action.motion_2d.dth
                    robot_pose = np.asarray(wait_for_robot_pose())
                    rx, ry, rz = robot_pose[:3]
                    thx, thy, thz = math_utils.quat_to_euler(*robot_pose[3:])
                    thz = (thz + angle) % 360
                    rx = rx + forward*math.cos(math_utils.to_rad(thz))
                    ry = ry + forward*math.sin(math_utils.to_rad(thz))
                    dest = (rx, ry, rz, *math_utils.euler_to_quat(thx, thy, thz))

                else:
                    raise NotImplementedError("Not implemented action.")

                nav_action = make_nav_action(dest[:3], dest[3:], goal_id=step)
                self._action_pub.publish(nav_action)
                rospy.loginfo("published nav action for execution")
                # wait for navigation done
                ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                          allow_headerless=True, verbose=True)
                rospy.loginfo("nav action done.")

            elif isinstance(action, a_pb2.Find):
                find_action = KeyValAction(stamp=rospy.Time.now(),
                                           type="find")
                self._action_pub.publish(find_action)
                rospy.loginfo("published find action for execution")
                ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                          allow_headerless=True, verbose=True)
                rospy.loginfo("find action done")

            rospy.loginfo("waiting for local agent creation...")
            local_robot_id = f"{self.robot_id}_local"
            self._sloop_client.waitForAgentCreation(local_robot_id)
            rospy.loginfo(f"local agent {local_robot_id} created!")

            # Now, wait for observation
            obs_msg = ros_utils.WaitForMessages([OBSERVATION_TOPIC],
                                                [KeyValObservation],
                                                verbose=True, allow_headerless=True).messages[0]
            detections_pb, robot_pose_pb, objects_found_pb =\
                observation_msg_to_proto(self.world_frame, obs_msg)

            # Now, send obseravtions for belief update
            header = proto_utils.make_header(frame_id=self.world_frame)
            response_observation = self._sloop_client.processObservation(
                self.robot_id, robot_pose_pb,
                object_detections=detections_pb,
                objects_found=objects_found_pb,
                header=header, return_fov=True,
                action_id=action_id, action_finished=True, debug=False)
            response_robot_belief = self._sloop_client.getRobotBelief(
                self.robot_id, header=proto_utils.make_header(self.world_frame))

            print(f"Step {step} robot belief:")
            robot_belief_pb = response_robot_belief.robot_belief
            objects_found = set(robot_belief_pb.objects_found.object_ids)
            print(f"  pose: {robot_belief_pb.pose.pose_3d}")
            print(f"  objects found: {objects_found}")
            print("-----------")

            # visualize FOV and belief
            if response_observation.HasField("fovs"):
                self.visualize_fovs_3d(response_observation)
            self.get_and_visualize_belief_3d(robot_id=local_robot_id)
            self.get_and_visualize_belief_2d()
            break

def main():
    TestSimpleEnvHierSearch(prior="uniform")

if __name__ == "__main__":
    main()
