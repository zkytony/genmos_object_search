#!/usr/bin/env python
# Test 2D local search in SimpleSimEnv
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
import math
import time
import rospy
import pickle
import json
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
from sloop_object_search.grpc.common_pb2 import Status
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils
from test_simple_sim_env_navigation import make_nav_action
from test_simple_sim_env_local_search_3d import (wait_for_robot_pose,
                                                 observation_msg_to_proto)

REGION_POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
INIT_ROBOT_POSE_TOPIC = "/simple_sim_env/init_robot_pose"
ACTION_TOPIC = "/simple_sim_env/pomdp_action"
ACTION_DONE_TOPIC = "/simple_sim_env/action_done"
OBSERVATION_TOPIC = "/simple_sim_env/pomdp_observation"

WORLD_FRAME = "graphnav_map"

SEARCH_SPACE_RESOLUTION = 0.15


import yaml
with open("./config_simple_sim_lab121_lidar.yaml") as f:
    CONFIG = yaml.safe_load(f)
    AGENT_CONFIG = CONFIG["agent_config"]
    TASK_CONFIG = CONFIG["task_config"]
    PLANNER_CONFIG = CONFIG["planner_config"]
    OBJECT_LOCATIONS = CONFIG["object_locations"]


class TestSimpleEnvLocalSearch2D:

    def get_and_visualize_belief(self):
        # First, clear existing belief messages
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._belief_2d_markers_pub.publish(clear_msg)

        response = self._sloop_client.getObjectBeliefs(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        rospy.loginfo("got belief")

        # visualize object belief
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            color = AGENT_CONFIG["objects"][bobj_pb.object_id].get(
                "color", [0.2, 0.7, 0.2])[:3]
            msg = ros_utils.make_object_belief2d_proto_markers_msg(
                bobj_pb, header, SEARCH_SPACE_RESOLUTION,
                color=color)
            markers.extend(msg.markers)
        self._belief_2d_markers_pub.publish(MarkerArray(markers))
        rospy.loginfo("belief visualized")


    def __init__(self, prior="uniform"):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node("test_simple_env_hier_search")

        # Initialize ROS stuff
        action_pub = rospy.Publisher(ACTION_TOPIC, KeyValAction, queue_size=10, latch=True)
        self._belief_2d_markers_pub = rospy.Publisher(
            "~belief_2d", MarkerArray, queue_size=10, latch=True)
        self._fovs_markers_pub = rospy.Publisher(
            "~fovs", MarkerArray, queue_size=10, latch=True)

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        self.agent_config = AGENT_CONFIG
        self.robot_id = AGENT_CONFIG["robot"]["id"]
        self.world_frame = WORLD_FRAME

        if prior == "groundtruth":
            AGENT_CONFIG["belief"]["prior"] = {}
            for objid in AGENT_CONFIG["targets"]:
                AGENT_CONFIG["belief"]["prior"][objid] = [[OBJECT_LOCATIONS[objid][:2], 0.99]]

        # First, create an agent
        self._sloop_client.createAgent(header=proto_utils.make_header(), config=AGENT_CONFIG,
                                       robot_id=self.robot_id)

        # need to get a region point cloud and a pose use that as search region
        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [REGION_POINT_CLOUD_TOPIC, INIT_ROBOT_POSE_TOPIC],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=10, verbose=True).messages
        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
        self._sloop_client.updateSearchRegion(header=cloud_pb.header,
                                              robot_id=self.robot_id,
                                              is_3d=False,
                                              robot_pose=robot_pose_pb,
                                              point_cloud=cloud_pb,
                                              search_region_params_2d={"layout_cut": 0.6,
                                                                       "region_size": 5.0,
                                                                       "brush_size": 0.5,
                                                                       "grid_size": SEARCH_SPACE_RESOLUTION,
                                                                       "debug": False})
        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        self.get_and_visualize_belief()

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

            if isinstance(action, a_pb2.MoveViewpoint):
                if action.HasField("motion_2d"):
                    forward = action.motion_2d.forward
                    angle = action.motion_2d.dth
                    robot_pose = np.asarray(wait_for_robot_pose())
                    rx, ry, rz = robot_pose[:3]
                    thx, thy, thz = math_utils.quat_to_euler(*robot_pose[3:])
                    thz = (thz + angle) % 360
                    rx = rx + forward*math.cos(math_utils.to_rad(thz))
                    ry = ry + forward*math.sin(math_utils.to_rad(thz))
                    dest = (rx, ry, rz, *math_utils.euler_to_quat(thx, thy, thz))
                nav_action = make_nav_action(dest[:3], dest[3:], goal_id=step)
                action_pub.publish(nav_action)
                rospy.loginfo("published nav action for execution")
                # wait for navigation done
                ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                          allow_headerless=True, verbose=True)
                rospy.loginfo("nav action done.")

            elif isinstance(action, a_pb2.Find):
                find_action = KeyValAction(stamp=rospy.Time.now(),
                                           type="find")
                action_pub.publish(find_action)
                rospy.loginfo("published find action for execution")
                ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                          allow_headerless=True, verbose=True)
                rospy.loginfo("find action done")

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

            self.get_and_visualize_belief()
            # Check if we are done
            if objects_found == set(AGENT_CONFIG["targets"]):
                rospy.loginfo("Done!")
                break
            time.sleep(1)



        rospy.spin()

def main():
    TestSimpleEnvLocalSearch2D(prior="groundtruth")

if __name__ == "__main__":
    main()
