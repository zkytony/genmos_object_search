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
from sloop_object_search.grpc.constants import Message
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils

SEARCH_SPACE_RESOLUTION_3D = 0.1
SEARCH_SPACE_RESOLUTION_2D = 0.3


class SloopMosROS:
    def __init__(self, name="sloop_ros"):
        self.name = name
        self._sloop_client = None

    def server_message_callback(self, message):
        if Message.match(message) == Message.REQUEST_LOCAL_SEARCH_REGION_UPDATE:
            local_robot_id = Message.forwhom(message)
            rospy.loginfo(f"will send a update search request to {local_robot_id}")
            self.update_search_region_3d(robot_id=local_robot_id)
            self._local_robot_id = local_robot_id
        elif Message.match(message) == Message.LOCAL_AGENT_REMOVED:
            local_robot_id = Message.forwhom(message)
            rospy.loginfo(f"local agent {local_robot_id} removed.")
            if local_robot_id != self._local_robot_id:
                rospy.logerr("removed local agent has an unexpected ID")
            self._local_robot_id = None

    def update_search_region_2d(self, robot_id=None):
        # need to get a region point cloud and a pose use that as search region
        if robot_id is None:
            robot_id = self.robot_id
        rospy.loginfo("Sending request to update search region (2D)")

        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [self._search_region_2d_topic, self._robot_pose_topic],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=10000, verbose=True).messages

        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

        search_region_config = self.agent_config.get("search_region", {}).get("2d", {})
        search_region_params_2d = dict(
            layout_cut=search_region_config.get("layout_cut", 0.6),
            grid_size=search_region_config.get("res", SEARCH_SPACE_RESOLUTION_2D),
            brush_size=search_region_config.get("brush_size", 0.5),
            debug=search_region_config.get("debug", False)
        )
        self._sloop_client.updateSearchRegion(
            header=cloud_pb.header,
            robot_id=robot_id,
            robot_pose=robot_pose_pb,
            point_cloud=cloud_pb,
            search_region_params_2d=search_region_params_2d)

    def update_search_region_3d(self, robot_id=None):
        if robot_id is None:
            robot_id = self.robot_id
        rospy.loginfo("Sending request to update search region (3D)")
        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [self._search_region_3d_topic, self._robot_pose_topic],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=100, verbose=True).messages
        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

        # parameters
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        search_region_params_3d = dict(
            octree_size=search_region_config.get("octree_size", 32),
            search_space_resolution=search_region_config.get("res", SEARCH_SPACE_RESOLUTION_3D),
            region_size_x=search_region_config.get("region_size_x", 4.0),
            region_size_y=search_region_config.get("region_size_y", 4.0),
            region_size_z=search_region_config.get("region_size_z", 2.0),
            debug=search_region_config.get("debug", False)
        )
        self._sloop_client.updateSearchRegion(
            header=cloud_pb.header,
            robot_id=robot_id,
            robot_pose=robot_pose_pb,
            point_cloud=cloud_pb,
            search_region_params_3d=search_region_params_3d)

    def update_search_region(self):
        if self.agent_config["agent_type"] == "local":
            if self.agent_config["agent_class"].endswith("3D"):
                self.update_search_region_3d()
            else:
                self.update_search_region_2d()
        elif self.agent_config["agent_type"] == "hierarchical":
            # local agent in hierarchical planning will get its search region
            # through server-client communication. Here, the client only needs
            # to send over the search region info for the global agent.
            self.update_search_region_2d()
        else:
            raise ValueError("Unexpected agent type: {}"\
                             .format(self.agent_config["agent_type"]))


    def main(self):
                 # o3dviz=False, prior="uniform",
                 # search_space_res_3d=SEARCH_SPACE_RESOLUTION_3D,
                 # search_space_res_2d=SEARCH_SPACE_RESOLUTION_2D):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node(self.name)

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        config = rospy.get_param("~config")  # access parameters together as a dictionary
        self.agent_config = config["agent_config"]
        self.robot_id = rospy.get_param("~robot_id")
        if self.robot_id != self.agent_config["robot"]["id"]:
            rospy.logwarn("robot id {} in rosparam overrides that in config {}"\
                          .format(self.robot_id, self.agent_config["robot"]["id"]))
            self.agent_config["robot"]["id"] = self.robot_id
        self.world_frame = rospy.get_param("~world_frame")

        # Initialize ROS stuff
        self._action_pub = rospy.Publisher(
            "~action", KeyValAction, queue_size=10, latch=True)
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

        # Remap these topics that we subscribe to, if needed.
        self._search_region_2d_point_cloud_topic = "~search_region_cloud_2d"
        self._search_region_3d_point_cloud_topic = "~search_region_cloud_3d"
        self._robot_pose_topic = "~robot_pose"

        # First, create an agent
        self._sloop_client.createAgent(
            header=proto_utils.make_header(), config=self.agent_config,
            robot_id=self.robot_id)

        # Make the client listen to server
        ls_future = self._sloop_client.listenToServer(
            self.robot_id, self.server_message_callback)
        self._local_robot_id = None  # needed if the planner is hierarchical

        # Update search region
