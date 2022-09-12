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
            [self._search_region_2d_point_cloud_topic, self._robot_pose_topic],
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

    @property
    def search_space_res_2d(self):
        search_region_config = self.agent_config.get("search_region", {}).get("2d", {})
        return search_region_config.get("res", SEARCH_SPACE_RESOLUTION_2D)

    @property
    def search_space_res_3d(self):
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        return search_region_config.get("res", SEARCH_SPACE_RESOLUTION_3D)

    def update_search_region_3d(self, robot_id=None):
        if robot_id is None:
            robot_id = self.robot_id
        rospy.loginfo("Sending request to update search region (3D)")
        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [self._search_region_3d_point_cloud_topic, self._robot_pose_topic],
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
                    voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros_utils.make_viz_marker_for_voxel(
                    voxel, header, color=hit_color, ns="fov",
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
            color = self.agent_config["objects"][bobj_pb.object_id].get(
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

    def get_and_visualize_belief(self):
        if self.agent_config["agent_type"] == "local":
            if self.agent_config["agent_class"].endswith("3D"):
                self.get_and_visualize_belief_3d()
            else:
                self.get_and_visualize_belief_2d()
        elif self.agent_config["agent_type"] == "hierarchical":
            # local agent in hierarchical planning will get its search region
            # through server-client communication. Here, the client only needs
            # to send over the search region info for the global agent.
            if self._local_robot_id is not None:
                self.get_and_visualize_belief_3d(robot_id=self._local_robot_id)
            self.get_and_visualize_belief_2d()
        else:
            raise ValueError("Unexpected agent type: {}"\
                             .format(self.agent_config["agent_type"]))


    def main(self):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node(self.name)

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        config = rospy.get_param("~config")  # access parameters together as a dictionary
        self.agent_config = config["agent_config"]
        self.planner_config = config["planner_config"]
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
        self.update_search_region()

        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief()

        # create planner
        response = self._sloop_client.createPlanner(config=self.planner_config,
                                                    header=proto_utils.make_header(),
                                                    robot_id=self.robot_id)
        rospy.loginfo("planner created!")
        rospy.spin()
