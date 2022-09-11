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

SEARCH_SPACE_RESOLUTION_3D = 0.1
SEARCH_SPACE_RESOLUTION_2D = 0.3


class SloopMosROS:
    def __init__(self, name="sloop_ros"):
        self.name = name
        self._sloop_client = None

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

        self.search_space_res_3d = rospy.get_param("search_res_3d", SEARCH_SPACE_RESOLUTION_3D)
        self.search_space_res_2d = rospy.get_param("search_res_2d", SEARCH_SPACE_RESOLUTION_2D)



        # First, create an agent
        self._sloop_client.createAgent(
            header=proto_utils.make_header(), config=self.agent_config,
            robot_id=self.robot_id)
