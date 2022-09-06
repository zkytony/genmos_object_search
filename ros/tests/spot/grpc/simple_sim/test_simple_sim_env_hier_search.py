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
        if Message.match(message) == Message.REQUEST_SEARCH_REGION_UPDATE:
            print("will send a update search request to the designated robot id")
            self.update_search_region_3d()

    def __init__(self, prior="uniform"):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node("test_simple_env_hier_search")

        # Initialize ROS stuff
        action_pub = rospy.Publisher(ACTION_TOPIC, KeyValAction, queue_size=10, latch=True)
        self._topo_map_2d_markers_pub = rospy.Publisher(
            "~topo_map_2d", MarkerArray, queue_size=10, latch=True)
        self._belief_2d_markers_pub = rospy.Publisher(
            "~belief_2d", MarkerArray, queue_size=10, latch=True)
        self._fovs_markers_pub = rospy.Publisher(
            "~fovs", MarkerArray, queue_size=10, latch=True)
        # because region cloud is latched and only published once (for now),
        # we'll need to save it for this test
        self._region_cloud_msg = None

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        self.agent_config = AGENT_CONFIG
        self.robot_id = AGENT_CONFIG["robot"]["id"]
        self.world_frame = WORLD_FRAME

        if prior == "groundtruth":
            AGENT_CONFIG["belief"]["prior"] = {}
            for objid in AGENT_CONFIG["targets"]:
                AGENT_CONFIG["belief"]["prior"][objid] = [[OBJECT_LOCATIONS[objid], 0.99]]

        # Make the client listen to server
        ls_future = self._sloop_client.listenToServer(self.robot_id, self.server_message_callback)

        # First, create an agent
        self._sloop_client.createAgent(header=proto_utils.make_header(), config=AGENT_CONFIG,
                                       robot_id=self.robot_id)
        self.update_search_region_2d()

        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief()

        # create planner
        response = self._sloop_client.createPlanner(config=PLANNER_CONFIG,
                                                    header=proto_utils.make_header(),
                                                    robot_id=self.robot_id)
        rospy.loginfo("planner created!")



        response_plan = self._sloop_client.planAction(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        action = proto_utils.interpret_planned_action(response_plan)
        action_id = response_plan.action_id
        rospy.loginfo("plan action finished. Action ID: {}".format(action_id))

        if action.name.startswith("stay"):
            print("Stay")
            self.update_search_region_3d()
            rospy.spin()
            # # Action is stay --> may need to create local search agent.
            # # Will send over an UpdateSearchRegion request to help that.
            # # need to get a region point cloud and a pose use that as search region


        # # Send planning requests
        # for step in range(TASK_CONFIG["max_steps"]):
        #     response_plan = self._sloop_client.planAction(
        #         self.robot_id, header=proto_utils.make_header(self.world_frame))
        #     action = proto_utils.interpret_planned_action(response_plan)
        #     action_id = response_plan.action_id
        #     rospy.loginfo("plan action finished. Action ID: {}".format(action_id))

        #     # Now, we need to execute the action, and receive observation
        #     # from SimpleEnv. First, convert the dest_3d in action to
        #     # a KeyValAction message
        #     if isinstance(action, a_pb2.MoveViewpoint):
        #         if action.HasField("dest_3d"):
        #             dest = proto_utils.poseproto_to_posetuple(action.dest_3d)
        #         elif action.HasField("dest_2d"):
        #             robot_pose = np.asarray(wait_for_robot_pose())
        #             dest_2d = proto_utils.poseproto_to_posetuple(action.dest_2d)
        #             x, y, thz = dest_2d
        #             z = robot_pose[2]
        #             thx, thy, _ = math_utils.quat_to_euler(*robot_pose[3:])
        #             dest = (x, y, z, *math_utils.euler_to_quat(thx, thy, thz))
        #         else:
        #             raise NotImplementedError("No relative motion right now.")

        #         nav_action = make_nav_action(dest[:3], dest[3:], goal_id=step)
        #         action_pub.publish(nav_action)
        #         rospy.loginfo("published nav action for execution")
        #         # wait for navigation done
        #         ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
        #                                   allow_headerless=True, verbose=True)
        #         rospy.loginfo("nav action done.")
        #     elif isinstance(action, a_pb2.Find):
        #         find_action = KeyValAction(stamp=rospy.Time.now(),
        #                                    type="find")
        #         action_pub.publish(find_action)
        #         rospy.loginfo("published find action for execution")
        #         ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
        #                                   allow_headerless=True, verbose=True)
        #         rospy.loginfo("find action done")

        #     # Now, wait for observation
        #     obs_msg = ros_utils.WaitForMessages([OBSERVATION_TOPIC],
        #                                         [KeyValObservation],
        #                                         verbose=True, allow_headerless=True).messages[0]
        #     detections_pb, robot_pose_pb, objects_found_pb =\
        #         observation_msg_to_proto(self.world_frame, obs_msg)

        #     # Now, send obseravtions for belief update
        #     header = proto_utils.make_header(frame_id=self.world_frame)
        #     response_observation = self._sloop_client.processObservation(
        #         self.robot_id, robot_pose_pb,
        #         object_detections=detections_pb,
        #         objects_found=objects_found_pb,
        #         header=header, return_fov=True,
        #         action_id=action_id, action_finished=True, debug=False)
        #     response_robot_belief = self._sloop_client.getRobotBelief(
        #         self.robot_id, header=proto_utils.make_header(self.world_frame))

        #     print(f"Step {step} robot belief:")
        #     robot_belief_pb = response_robot_belief.robot_belief
        #     objects_found = set(robot_belief_pb.objects_found.object_ids)
        #     print(f"  pose: {robot_belief_pb.pose.pose_3d}")
        #     print(f"  objects found: {objects_found}")
        #     print("-----------")

        #     self.get_and_visualize_belief()
        #     # Check if we are done
        #     if objects_found == set(AGENT_CONFIG["targets"]):
        #         rospy.loginfo("Done!")
        #         break
        #     time.sleep(1)

def main():
    TestSimpleEnvHierSearch(prior="uniform")

if __name__ == "__main__":
    main()
