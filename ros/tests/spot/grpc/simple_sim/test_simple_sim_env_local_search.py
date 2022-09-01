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


REGION_POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
INIT_ROBOT_POSE_TOPIC = "/simple_sim_env/init_robot_pose"
ACTION_TOPIC = "/simple_sim_env/pomdp_action"
ACTION_DONE_TOPIC = "/simple_sim_env/action_done"
OBSERVATION_TOPIC = "/simple_sim_env/pomdp_observation"

WORLD_FRAME = "graphnav_map"


import yaml
with open("./config_simple_sim_lab121_lidar.yaml") as f:
    CONFIG = yaml.safe_load(f)
    AGENT_CONFIG = CONFIG["agent_config"]
    TASK_CONFIG = CONFIG["task_config"]
    PLANNER_CONFIG = CONFIG["planner_config"]
    OBJECT_LOCATIONS = CONFIG["object_locations"]

def observation_msg_to_proto(world_frame, o_msg):
    """returns three observation proto objects: (ObjectDetectionArray, RobotPose,
    ObjectsFound) This is reasonable because it's not typically the case that
    you receive all observations as a joint KeyValObservation message.
    """
    if o_msg.type != "joint":
        raise NotImplementedError(f"Cannot handle type {o_msg.type}")

    header = proto_utils.make_header(frame_id=world_frame)
    kv = {k:v for k,v in zip(o_msg.keys, o_msg.values)}

    robot_id = kv["robot_id"]
    robot_pose = eval(kv["robot_pose"])
    objects_found = eval(kv["objects_found"])
    robot_pose_pb = o_pb2.RobotPose(header=header, robot_id=robot_id,
                                    pose_3d=proto_utils.posetuple_to_poseproto(robot_pose))
    objects_found_pb = o_pb2.ObjectsFound(header=header, robot_id=robot_id,
                                          object_ids=objects_found)

    # figure out what objects there are
    object_ids = set()
    for k in kv:
        if k.startswith("loc"):
            object_ids.add(k.split("_")[1])

    detections = []
    for objid in object_ids:
        objloc = eval(kv[f"loc_{objid}"])
        objsizes = eval(kv[f"sizes_{objid}"])
        if objloc is not None:
            objbox = common_pb2.Box3D(center=proto_utils.posetuple_to_poseproto((*objloc, 0, 0, 0, 1)),
                                      sizes=common_pb2.Vec3(x=objsizes[0], y=objsizes[1], z=objsizes[2]))
            detections.append(o_pb2.Detection3D(label=objid, box=objbox))
    detections_pb = o_pb2.ObjectDetectionArray(header=header,
                                               robot_id=robot_id,
                                               detections=detections)
    return detections_pb, robot_pose_pb, objects_found_pb

def wait_for_robot_pose():
    obs_msg = ros_utils.WaitForMessages([OBSERVATION_TOPIC],
                                        [KeyValObservation],
                                        verbose=True, allow_headerless=True).messages[0]
    kv = {k:v for k,v in zip(obs_msg.keys, obs_msg.values)}
    robot_pose = eval(kv["robot_pose"])
    return robot_pose

class TestSimpleEnvLocalSearch:

    def visualize_fovs(self, response):
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        fovs = json.loads(response.fovs.decode('utf-8'))
        markers = []
        for objid in fovs:
            free_color = np.array(self.agent_config["objects"][objid].get(
                "color", [0.8, 0.4, 0.8]))[:3]
            hit_color = lighter(free_color*255, -0.25)/255

            obstacles_hit = set(map(tuple, fovs[objid]['obstacles_hit']))
            for voxel in fovs[objid]['visible_volume']:
                voxel = tuple(voxel)
                if voxel in obstacles_hit:
                    continue
                m = ros_utils.make_viz_marker_for_voxel(
                    voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros_utils.make_viz_marker_for_voxel(
                    voxel, header, color=hit_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
        self._fovs_markers_pub.publish(MarkerArray(markers))

    def get_and_visualize_belief(self, o3dviz=True):
        response = self._sloop_client.getObjectBeliefs(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        rospy.loginfo("got belief")

        # visualize the belief
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            msg = ros_utils.make_octree_belief_proto_markers_msg(
                bobj_pb, header, alpha_scaling=20.0)
            markers.extend(msg.markers)
        self._octbelief_markers_pub.publish(MarkerArray(markers))

        rospy.loginfo("belief visualized")


    def __init__(self, o3dviz=False, prior="uniform"):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node("test_simple_env_local_search")

        # Initialize ROS stuff
        action_pub = rospy.Publisher(ACTION_TOPIC, KeyValAction, queue_size=10, latch=True)
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)
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
                AGENT_CONFIG["belief"]["prior"][objid] = [[OBJECT_LOCATIONS[objid], 0.99]]

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
                                              is_3d=True,
                                              robot_pose=robot_pose_pb,
                                              point_cloud=cloud_pb,
                                              search_region_params_3d={"octree_size": 32,
                                                                       "search_space_resolution": 0.1,
                                                                       "debug": False,
                                                                       "region_size_x": 4.0,
                                                                       "region_size_y": 4.0,
                                                                       "region_size_z": 2.5})
        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief(o3dviz=o3dviz)

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

            # Clear markers
            header = std_msgs.Header(stamp=rospy.Time.now(),
                                     frame_id=self.world_frame)
            clear_msg = ros_utils.clear_markers(header, ns="")
            self._octbelief_markers_pub.publish(clear_msg)
            self._fovs_markers_pub.publish(clear_msg)

            # visualize FOV and belief
            self.visualize_fovs(response_observation)
            self.get_and_visualize_belief(o3dviz=o3dviz)

            # Check if we are done
            if objects_found == set(AGENT_CONFIG["targets"]):
                rospy.loginfo("Done!")
                break
            time.sleep(1)



def main():
    TestSimpleEnvLocalSearch(o3dviz=False, prior="groundtruth")

if __name__ == "__main__":
    main()
