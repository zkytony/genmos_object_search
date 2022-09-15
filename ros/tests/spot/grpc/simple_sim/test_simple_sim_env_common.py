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

WORLD_FRAME = "graphnav_map"

REGION_POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
MAP_POINT_CLOUD_TOPIC = "/graphnav_map_publisher/graphnav_points"
INIT_ROBOT_POSE_TOPIC = "/simple_sim_env/init_robot_pose"
ROBOT_POSE_TOPIC = "/simple_sim_env/robot_pose"
ACTION_TOPIC = "/simple_sim_env/pomdp_action"
RESET_TOPIC = "/simple_sim_env/reset"
ACTION_DONE_TOPIC = "/simple_sim_env/action_done"
OBSERVATION_TOPIC = "/simple_sim_env/pomdp_observation"

SEARCH_SPACE_RESOLUTION_3D = 0.1
SEARCH_SPACE_RESOLUTION_2D = 0.3


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


class TestSimpleEnvCase:

    def visualize_fovs_3d(self, response):
        # Clear markers
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._fovs_markers_pub.publish(clear_msg)

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
                    objid, voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=hit_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
        self._fovs_markers_pub.publish(MarkerArray(markers))

    def get_and_visualize_belief_3d(self, robot_id=None, o3dviz=True):
        if robot_id is None:
            robot_id = self.robot_id

        # Clear markers
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._octbelief_markers_pub.publish(clear_msg)
        self._topo_map_3d_markers_pub.publish(clear_msg)

        response = self._sloop_client.getObjectBeliefs(
            robot_id, header=proto_utils.make_header(self.world_frame))
        if response.status != Status.SUCCESSFUL:
            print("Failed to get 3D belief")
            return
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

        # visualize topo map in robot belief
        markers = []
        response_robot_belief = self._sloop_client.getRobotBelief(
            robot_id, header=proto_utils.make_header(self.world_frame))
        robot_belief_pb = response_robot_belief.robot_belief
        if robot_belief_pb.HasField("topo_map"):
            msg = ros_utils.make_topo_map_proto_markers_msg(
                robot_belief_pb.topo_map,
                header, self.search_space_res_3d,
                node_color=[0.82, 0.01, 0.08, 0.8],
                edge_color=[0.24, 0.82, 0.01, 0.8],
                node_thickness=self.search_space_res_3d)
            markers.extend(msg.markers)
        self._topo_map_3d_markers_pub.publish(MarkerArray(markers))
        rospy.loginfo("belief visualized")

    def get_and_visualize_belief_2d(self):
        # First, clear existing belief messages
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._belief_2d_markers_pub.publish(clear_msg)
        self._topo_map_2d_markers_pub.publish(clear_msg)

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
                bobj_pb, header, self.search_space_res_2d,
                color=color)
            markers.extend(msg.markers)
        self._belief_2d_markers_pub.publish(MarkerArray(markers))

        # visualize topo map in robot belief
        markers = []
        response_robot_belief = self._sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        robot_belief_pb = response_robot_belief.robot_belief
        if robot_belief_pb.HasField("topo_map"):
            msg = ros_utils.make_topo_map_proto_markers_msg(
                robot_belief_pb.topo_map,
                header, self.search_space_res_2d)
            markers.extend(msg.markers)
        self._topo_map_2d_markers_pub.publish(MarkerArray(markers))
        rospy.loginfo("belief visualized")

    def update_search_region_2d(self, robot_id=None):
        # need to get a region point cloud and a pose use that as search region
        if robot_id is None:
            robot_id = self.robot_id
        rospy.loginfo("Sending request to update search region (2D)")

        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [MAP_POINT_CLOUD_TOPIC, ROBOT_POSE_TOPIC],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=10000, verbose=True).messages

        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
        self._sloop_client.updateSearchRegion(header=cloud_pb.header,
                                              robot_id=robot_id,
                                              robot_pose=robot_pose_pb,
                                              point_cloud=cloud_pb,
                                              search_region_params_2d={"layout_cut": 0.6,
                                                                       "region_size": 15.0,
                                                                       "brush_size": 0.5,
                                                                       "grid_size": self.search_space_res_2d,
                                                                       "debug": False})

    def update_search_region_3d(self, robot_id=None):
        if robot_id is None:
            robot_id = self.robot_id
        rospy.loginfo("Sending request to update search region (3D)")
        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [REGION_POINT_CLOUD_TOPIC, ROBOT_POSE_TOPIC],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=100, verbose=True).messages
        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
        self._sloop_client.updateSearchRegion(header=cloud_pb.header,
                                              robot_id=robot_id,
                                              robot_pose=robot_pose_pb,
                                              point_cloud=cloud_pb,
                                              search_region_params_3d={"octree_size": 64,
                                                                       "search_space_resolution": self.search_space_res_3d,
                                                                       "debug": False,
                                                                       "region_size_x": 4.0,
                                                                       "region_size_y": 4.0,
                                                                       "region_size_z": 2.4})

    def __init__(self, name="test_simple_env_search",
                 o3dviz=False, prior="uniform",
                 search_space_res_3d=SEARCH_SPACE_RESOLUTION_3D,
                 search_space_res_2d=SEARCH_SPACE_RESOLUTION_2D):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node(name)

        # Initialize ROS stuff
        self._action_pub = rospy.Publisher(ACTION_TOPIC, KeyValAction, queue_size=10, latch=True)
        self._reset_pub = rospy.Publisher(RESET_TOPIC, std_msgs.String, queue_size=10)
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)
        self._fovs_markers_pub = rospy.Publisher(
            "~fovs", MarkerArray, queue_size=10, latch=True)
        self._topo_map_3d_markers_pub = rospy.Publisher(
            "~topo_map_3d", MarkerArray, queue_size=10, latch=True)
        self._topo_map_2d_markers_pub = rospy.Publisher(
            "~topo_map_2d", MarkerArray, queue_size=10, latch=True)
        self._belief_2d_markers_pub = rospy.Publisher(
            "~belief_2d", MarkerArray, queue_size=10, latch=True)

        self.search_space_res_3d = search_space_res_3d
        self.search_space_res_2d = search_space_res_2d

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        self.agent_config = AGENT_CONFIG
        self.robot_id = AGENT_CONFIG["robot"]["id"]
        self.world_frame = WORLD_FRAME

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

        self._region_cloud_msg = None
        # the rest is determined by the child class

    def reset(self):
        self._sloop_client.reset()
        rospy.loginfo("Server reset done.")
        self._reset_pub.publish(std_msgs.String("reset"))
        time.sleep(0.5)
        rospy.loginfo("Simple env reset done.")
