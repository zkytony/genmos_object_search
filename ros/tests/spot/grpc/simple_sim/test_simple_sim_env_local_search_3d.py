#!/usr/bin/env python
# Test 3D local search in SimpleSimEnv.
#
# To run the test, do the following IN ORDER:
# 0. run config_simple_sim_lab121_lidar.py to generate the .yaml configuration file
# 1. run in a terminal 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=lab121_lidar'
# 2. run in a terminal 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose'
# 3. run in a terminal 'roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=lab121_lidar'
# 4. run in a terminal 'python -m sloop_object_search.grpc.server'
# 5. run in a terminal 'python test_simple_sim_env_local_search.py'
# 6. run in a terminal 'roslaunch sloop_object_search_ros view_simple_sim.launch'
# 7. to monitor CPU temperature: 'watch -n 1 -x sensors'
# ------------------
#
# We are testing the local search algorithm. We need to do:
# - Specify a scenario (object poses, robot starting pose).
# - Create an agent (rpc).
# - Visualize agent's belief
# - When the agent is ready, send planning request
# - Execute planned action
# - Synthesize observation
# - Visualize observation (e.g. FOV)
# - Update belief
# - Consider a large viewpoint-based action space.
# - Consider correlation
# - Consider realistic camera parameters
#
# Remember, you are a USER of the sloop_object_search package.
# Not its developer. You should only need to do basic things.
import numpy as np
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
from test_simple_sim_env_common import (TestSimpleEnvCase,
                                        observation_msg_to_proto,
                                        wait_for_robot_pose,
                                        REGION_POINT_CLOUD_TOPIC,
                                        INIT_ROBOT_POSE_TOPIC,
                                        ACTION_DONE_TOPIC, OBSERVATION_TOPIC,
                                        AGENT_CONFIG, PLANNER_CONFIG,
                                        TASK_CONFIG)



class TestSimpleEnvLocalSearch(TestSimpleEnvCase):

    def __init__(self, o3dviz=False, prior="uniform"):
        super().__init__(o3dviz=o3dviz, prior=prior)

        self.update_search_region_3d()

        self.report = {"steps": [], "total_time": 0, "success": False}

        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief_3d(o3dviz=o3dviz)

        # create planner
        response = self._sloop_client.createPlanner(config=PLANNER_CONFIG,
                                                    header=proto_utils.make_header(),
                                                    robot_id=self.robot_id)
        rospy.loginfo("planner created!")

        _start_time = time.time()

        # Send planning requests
        for step in range(TASK_CONFIG["max_steps"]):
            _time = time.time()
            response_plan = self._sloop_client.planAction(
                self.robot_id, header=proto_utils.make_header(self.world_frame))
            _planning_time = time.time() - _time

            _time = time.time()
            action = proto_utils.interpret_planned_action(response_plan)
            action_id = response_plan.action_id
            rospy.loginfo("plan action finished. Action ID: {}".format(action_id))

            # Now, we need to execute the action, and receive observation
            # from SimpleEnv. First, convert the dest_3d in action to
            # a KeyValAction message
            if isinstance(action, a_pb2.MoveViewpoint):
                if action.HasField("dest_3d"):
                    dest = proto_utils.poseproto_to_posetuple(action.dest_3d)
                else:
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
            _action_time = time.time() - _time

            # Now, wait for observation
            _time = time.time()
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
            _observation_and_belief_update_time = time.time() - _time

            response_robot_belief = self._sloop_client.getRobotBelief(
                self.robot_id, header=proto_utils.make_header(self.world_frame))

            print(f"Step {step} robot belief:")
            robot_belief_pb = response_robot_belief.robot_belief
            objects_found = set(robot_belief_pb.objects_found.object_ids)
            print(f"  pose: {robot_belief_pb.pose.pose_3d}")
            print(f"  objects found: {objects_found}")
            print("-----------")

            # Report
            self.report["steps"].append({"robot_pose": proto_utils.robot_pose_from_proto(robot_pose_pb),
                                         "planning_time": _planning_time})

            # visualize FOV and belief
            self.visualize_fovs_3d(response_observation)
            self.get_and_visualize_belief_3d(o3dviz=o3dviz)

            # Check if we are done
            if objects_found == set(AGENT_CONFIG["targets"]):
                rospy.loginfo("Done!")
                self.report["success"] = True
                break
            self.report["total_time"] += _planning_time + _action_time + _observation_and_belief_update_time
            if self.report["total_time"] > TASK_CONFIG.get("max_time", float('inf')):
                rospy.loginfo("Time out!")
                break
            time.sleep(1)
        self.report["total_time"] = time.time() - _start_time

import os
import pandas as pd
import datetime
def save_report(name, report):
    os.makedirs("./report", exist_ok=True)
    length = 0
    for i in range(1, len(report["steps"])):
        prev_s = report["steps"][i-1]
        s = report["steps"][i]
        length += math_utils.euclidean_dist(
            prev_s["robot_pose"][:3], s["robot_pose"][:3])

    planning_time = 0
    for i in range(len(report["steps"])):
        planning_time += report["steps"][i]["planning_time"]

    ct = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    df = pd.DataFrame({"name": name,
                       "timestamp": ct,
                       "length": length,
                       "planning_time": planning_time,
                       "total_time": report["total_time"],
                       "success": report["success"]})
    df.to_csv(os.path.join("report", f"report_{name}_{ct}.csv"))


def main():
    test = TestSimpleEnvLocalSearch(o3dviz=False, prior="groundtruth")
    test.reset()
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    test = TestSimpleEnvLocalSearch(o3dviz=False, prior="groundtruth")

if __name__ == "__main__":
    main()
