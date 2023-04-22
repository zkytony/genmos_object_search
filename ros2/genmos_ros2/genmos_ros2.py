# to test:
# ros2 launch genmos_object_search_ros2 run_search_client.launch config_file:=src/genmos_object_search_ros2/tests/simple_sim/config_simple_sim_lab121_lidar.yaml
import rclpy
import tf2_ros
import math
import numpy as np
import time
import pickle
import yaml
import json
import threading
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import vision_msgs.msg as vision_msgs
from visualization_msgs.msg import Marker, MarkerArray

from genmos_object_search_ros2.msg import KeyValAction, KeyValObservation
from genmos_object_search.grpc.client import GenMOSClient
from genmos_object_search.grpc.utils import proto_utils
from genmos_object_search.utils.open3d_utils import draw_octree_dist
from genmos_object_search.grpc import genmos_object_search_pb2 as gmpb2
from genmos_object_search.grpc import observation_pb2 as o_pb2
from genmos_object_search.grpc import action_pb2 as a_pb2
from genmos_object_search.grpc import common_pb2
from genmos_object_search.grpc.common_pb2 import Status
from genmos_object_search.grpc.constants import Message
from genmos_object_search.utils.colors import lighter
from genmos_object_search.utils import math as math_utils
from genmos_object_search.utils.misc import import_class
from genmos_object_search.utils import typ
from . import ros2_utils


SEARCH_SPACE_RESOLUTION_3D = 0.1
SEARCH_SPACE_RESOLUTION_2D = 0.3


class GenMOSROS2(Node):
    """This node essentially funnels sensor messages from
    ROS2 to GenMOS and processes planned actions to be
    executed on the robot through ROS2, and manages the
    running of the search task.

    Note that this Node does not constantly subscribes
    to any topic, but only reads messages from certain
    topics from time to time based on need, through the
    blocking ros2_utils.wait_for_messages call.

    Subscribes:
      ~/search_region_cloud_2d (sensor_msgs.PointCloud2)
      ~/search_region_cloud_3d (sensor_msgs.PointCloud2)
      ~/search_region_center (geometry_msgs.PoseStamped)
      ~/robot_pose (geometry_msgs.PoseStamped)
      ~/object_detections (geometry_msgs.PoseStamped)
      ~/action_done (std_msgs.String)

    Publishes:
      ~/action (KeyValAction)
      ~/octree_belief (MarkerArray)
      ~/fovs (MarkerArray)
      ~/topo_map_3d (MarkerArray)
      ~/topo_map_2d (MarkerArray)
      ~/belief_2d (MarkerArray)

    """
    def __init__(self, name="genmos_ros2", verbose=True):
        super().__init__(name)

        # ROS2 Stuff
        ## Parameters
        params = [("robot_id", "robot0"),
                  ("world_frame", "graphnav_map"),
                  ("config_file", ""),
                  ("obs_queue_size", 200),
                  ("obs_delay", 1.0),
                  ("dynamic_update", False)]
        param_names = [p[0] for p in params]
        ros2_utils.declare_params(self, params)
        ros2_utils.print_parameters(self, param_names)

        self.robot_id = self.get_parameter("robot_id").value
        self.world_frame = self.get_parameter("world_frame").value
        self.obqueue_size = self.get_parameter("obs_queue_size").value
        self.obdelay = self.get_parameter("obs_delay").value
        self.dynamic_update = self.get_parameter("dynamic_update").value

        ## Publishers
        self._action_pub = self.create_publisher(
            KeyValAction, "~/action", ros2_utils.latch(depth=10))
        self._octbelief_markers_pub = self.create_publisher(
            MarkerArray, "~/octree_belief", ros2_utils.latch(depth=10))
        self._fovs_markers_pub = self.create_publisher(
            MarkerArray, "~/fovs", ros2_utils.latch(depth=10))
        self._topo_map_3d_markers_pub = self.create_publisher(
            MarkerArray, "~/topo_map_3d", ros2_utils.latch(depth=10))
        self._topo_map_2d_markers_pub = self.create_publisher(
            MarkerArray, "~/topo_map_2d", ros2_utils.latch(depth=10))
        self._belief_2d_markers_pub = self.create_publisher(
            MarkerArray, "~/belief_2d", ros2_utils.latch(depth=10))

        ## callback groups
        self.wfm_cb_group = MutuallyExclusiveCallbackGroup()
        self.adhoc_cb_group = MutuallyExclusiveCallbackGroup()
        self.periodic_cb_group = MutuallyExclusiveCallbackGroup()

        ## Topics for subscriptions
        self._search_region_2d_point_cloud_topic = "~/search_region_cloud_2d"
        self._search_region_3d_point_cloud_topic = "~/search_region_cloud_3d"
        self._search_region_center_topic = "~/search_region_center"
        self._robot_pose_topic = "~/robot_pose"
        ## Note for object detections, we accept 3D bounding-box-based detections.
        self._object_detections_topic = "~/object_detections"
        self._action_done_topic = "~/action_done"

        # tf; need to create listener early enough before looking up to let tf propagate into buffer
        # reference: https://answers.ros.org/question/292096/right_arm_base_link-passed-to-lookuptransform-argument-target_frame-does-not-exist/
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer, self)

        # GenMOS Client
        ## Initialize grpc client
        self._genmos_client = GenMOSClient()

        ## Loading configuration
        config_file = self.get_parameter("config_file").value
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.config = config
        self.agent_config = config["agent_config"]
        self.planner_config = config["planner_config"]
        if self.robot_id != self.agent_config["robot"]["id"]:
            self.get_logger().warning("robot id {} in rosparam overrides that in config {}"\
                                      .format(self.robot_id, self.agent_config["robot"]["id"]))
            self.agent_config["robot"]["id"] = self.robot_id
        ## object detector model's output class names
        self.detection_class_names = self.config["ros2"]["detection_class_names"]

        # internal states
        self.last_action = None
        self.objects_found = set()


    def server_message_callback(self, message):
        if Message.match(message) == Message.REQUEST_LOCAL_SEARCH_REGION_UPDATE:
            local_robot_id = Message.forwhom(message)
            self.get_logger().info(f"will send a update search request to {local_robot_id}")
            self.update_search_region_3d(robot_id=local_robot_id)
            self._local_robot_id = local_robot_id
        elif Message.match(message) == Message.LOCAL_AGENT_REMOVED:
            local_robot_id = Message.forwhom(message)
            self.get_logger().info(f"local agent {local_robot_id} removed.")
            if local_robot_id != self._local_robot_id:
                self.get_logger().error("removed local agent has an unexpected ID")
            self._local_robot_id = None

#     def update_search_region_2d(self, robot_id=None):
#         # need to get a region point cloud and a pose use that as search region
#         if robot_id is None:
#             robot_id = self.robot_id
#         self.get_logger().info("Sending request to update search region (2D)")

#         region_cloud_msg, pose_stamped_msg = ros2_utils.WaitForMessages(
#             [self._search_region_2d_point_cloud_topic, self._search_region_center_topic],
#             [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
#             delay=10000, verbose=True).messages

#         cloud_pb = ros2_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
#         robot_pose = ros2_utils.pose_to_tuple(pose_stamped_msg.pose)
#         robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

#         search_region_config = self.agent_config.get("search_region", {}).get("2d", {})
#         search_region_params_2d = dict(
#             layout_cut=search_region_config.get("layout_cut", 0.6),
#             grid_size=search_region_config.get("res", SEARCH_SPACE_RESOLUTION_2D),
#             region_size=search_region_config.get("region_size", 3.0),
#             brush_size=search_region_config.get("brush_size", 0.5),
#             include_free=search_region_config.get("include_free", True),
#             include_obstacles=search_region_config.get("include_obstacles", False),
#             expansion_width=search_region_config.get("expansion_width", 0.5),
#             debug=search_region_config.get("debug", False)
#         )
#         self._genmos_client.updateSearchRegion(
#             header=cloud_pb.header,
#             robot_id=robot_id,
#             robot_pose=robot_pose_pb,
#             point_cloud=cloud_pb,
#             search_region_params_2d=search_region_params_2d)

#     @property
#     def search_space_res_2d(self):
#         search_region_config = self.agent_config.get("search_region", {}).get("2d", {})
#         return search_region_config.get("res", SEARCH_SPACE_RESOLUTION_2D)

    @property
    def search_space_res_3d(self):
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        return search_region_config.get("res", SEARCH_SPACE_RESOLUTION_3D)

    def update_search_region_3d(self, robot_id=None):
        if robot_id is None:
            robot_id = self.robot_id
        self.get_logger().info("Sending request to update search region (3D)")
        region_cloud_msg, pose_stamped_msg = ros2_utils.wait_for_messages(
            self, [self._search_region_3d_point_cloud_topic, self._search_region_center_topic],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=100, verbose=True)
        cloud_pb = ros2_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose = ros2_utils.pose_to_tuple(pose_stamped_msg.pose)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

        # parameters
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        search_region_params_3d = dict(
            octree_size=search_region_config.get("octree_size", 32),
            search_space_resolution=search_region_config.get("res", SEARCH_SPACE_RESOLUTION_3D),
            region_size_x=search_region_config.get("region_size_x"),
            region_size_y=search_region_config.get("region_size_y"),
            region_size_z=search_region_config.get("region_size_z"),
            debug=search_region_config.get("debug", False)
        )
        self._genmos_client.updateSearchRegion(
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

    def clear_fovs_markers(self):
        # Clear markers
        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        clear_msg = ros2_utils.clear_markers(header, ns="")
        self._fovs_markers_pub.publish(clear_msg)

    def clear_octree_markers(self):
        # Clear markers
        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        clear_msg = ros2_utils.clear_markers(header, ns="")
        self._octbelief_markers_pub.publish(clear_msg)

    def visualize_fovs_3d(self, response):
        # Clear markers
        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        clear_msg = ros2_utils.clear_markers(header, ns="")
        self._fovs_markers_pub.publish(clear_msg)

        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        fovs = json.loads(response.fovs.decode('utf-8'))
        markers = []
        for objid in fovs:
            if objid in self.objects_found:
                continue
            free_color = np.array(self.agent_config["objects"][objid].get(
                "color", [0.8, 0.4, 0.8]))[:3]
            hit_color = lighter(free_color*255, -0.25)/255

            obstacles_hit = set(map(tuple, fovs[objid]['obstacles_hit']))
            for voxel in fovs[objid]['visible_volume']:
                voxel = tuple(voxel)
                if voxel in obstacles_hit:
                    continue
                m = ros2_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros2_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=hit_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            break  # just visualize one
        self._fovs_markers_pub.publish(MarkerArray(markers=markers))

    def get_and_visualize_belief_3d(self, robot_id=None, o3dviz=True):
        if robot_id is None:
            robot_id = self.robot_id

        # Clear markers
        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        clear_msg = ros2_utils.clear_markers(header, ns="")
        self._octbelief_markers_pub.publish(clear_msg)
        self._topo_map_3d_markers_pub.publish(clear_msg)

        response = self._genmos_client.getObjectBeliefs(
            robot_id, header=proto_utils.make_header(self.world_frame))
        if response.status != Status.SUCCESSFUL:
            print("Failed to get 3D belief")
            return
        self.get_logger().info("got belief")

        # visualize the belief
        header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                 frame_id=self.world_frame)
        markers = []
        # First, visualize the belief of detected objects
        for bobj_pb in response.object_beliefs:
            if bobj_pb.object_id in self.objects_found:
                msg = ros2_utils.make_octree_belief_proto_markers_msg(
                    bobj_pb, header, alpha_scaling=2.0, prob_thres=0.5)
                markers.extend(msg.markers)
        # For the other objects, just visualize one is enough.
        for bobj_pb in response.object_beliefs:
            if bobj_pb.object_id not in self.objects_found:
                msg = ros2_utils.make_octree_belief_proto_markers_msg(
                    bobj_pb, header, alpha_scaling=1.0)
                markers.extend(msg.markers)
                break
        self._octbelief_markers_pub.publish(MarkerArray(markers=markers))

        self.get_logger().info("belief visualized")

        # visualize topo map in robot belief
        markers = []
        response_robot_belief = self._genmos_client.getRobotBelief(
            robot_id, header=proto_utils.make_header(self.world_frame))
        robot_belief_pb = response_robot_belief.robot_belief
        if robot_belief_pb.HasField("topo_map"):
            msg = ros2_utils.make_topo_map_proto_markers_msg(
                robot_belief_pb.topo_map,
                header, self.search_space_res_3d,
                node_color=[0.82, 0.01, 0.08, 0.8],
                edge_color=[0.24, 0.82, 0.01, 0.8],
                node_thickness=self.search_space_res_3d)
            markers.extend(msg.markers)
        self._topo_map_3d_markers_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info("belief visualized")

#     def get_and_visualize_belief_2d(self):
#         # First, clear existing belief messages
#         header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
#                                  frame_id=self.world_frame)
#         clear_msg = ros2_utils.clear_markers(header, ns="")
#         self._belief_2d_markers_pub.publish(clear_msg)
#         self._topo_map_2d_markers_pub.publish(clear_msg)

#         response = self._genmos_client.getObjectBeliefs(
#             self.robot_id, header=proto_utils.make_header(self.world_frame))
#         assert response.status == Status.SUCCESSFUL
#         self.get_logger().info("got belief")

#         # height for 2d markers
#         _pos_z = self.ros_visual_config.get("marker2d_z", 0.1)

#         # visualize object belief
#         header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
#                                  frame_id=self.world_frame)
#         markers = []
#         for bobj_pb in response.object_beliefs:
#             color = self.agent_config["objects"][bobj_pb.object_id].get(
#                 "color", [0.2, 0.7, 0.2])[:3]
#             msg = ros2_utils.make_object_belief2d_proto_markers_msg(
#                 bobj_pb, header, self.search_space_res_2d,
#                 color=color, pos_z=_pos_z)
#             markers.extend(msg.markers)
#             if bobj_pb.object_id not in self.objects_found:
#                 break  # just visualize one, unless it's found
#         self._belief_2d_markers_pub.publish(MarkerArray(markers=markers))

#         # visualize topo map in robot belief
#         markers = []
#         response_robot_belief = self._genmos_client.getRobotBelief(
#             self.robot_id, header=proto_utils.make_header(self.world_frame))
#         robot_belief_pb = response_robot_belief.robot_belief
#         if robot_belief_pb.HasField("topo_map"):
#             msg = ros2_utils.make_topo_map_proto_markers_msg(
#                 robot_belief_pb.topo_map,
#                 header, self.search_space_res_2d,
#                 node_thickness=0.05,
#                 pos_z=_pos_z + 0.05)
#             markers.extend(msg.markers)
#         self._topo_map_2d_markers_pub.publish(MarkerArray(markers=markers))
#         self.get_logger().info("belief visualized")

    def get_and_visualize_belief(self):
        if self.agent_config["agent_type"] == "local":
            if self.agent_config["agent_class"].endswith("3D"):
                self.get_and_visualize_belief_3d()
            else:
                header = std_msgs.Header(stamp=self.get_clock().now().to_msg(),
                                         frame_id=self.world_frame)
                clear_msg = ros2_utils.clear_markers(header, ns="")
                self._fovs_markers_pub.publish(clear_msg)
                self._octbelief_markers_pub.publish(clear_msg)
                self._topo_map_3d_markers_pub.publish(clear_msg)
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

    def wait_for_observation(self):
        """We wait for the robot pose (PoseStamped) and the
        object detections (vision_msgs.Detection3DArray)

        Returns:
            a tuple: (detections_pb, robot_pose_pb, objects_found_pb)"""
        # robot pose may be much higher in frequency than object detection.
        robot_pose_msg, object_detections_msg = ros2_utils.wait_for_messages(
            self, [self._robot_pose_topic, self._object_detections_topic],
            [geometry_msgs.PoseStamped, vision_msgs.Detection3DArray],
            queue_size=self.obqueue_size, delay=self.obdelay, verbose=True,
            callback_group=self.wfm_cb_group)

        if robot_pose_msg.header.frame_id != self.world_frame:
            # Need to convert robot pose to world frame
            robot_pose_msg = ros2_utils.tf2_transform(self.tfbuffer, robot_pose_msg, self.world_frame)

        # Detection proto
        detections_pb = ros2_utils.detection3darray_to_proto(
            object_detections_msg, self.robot_id, self.detection_class_names,
            target_frame=self.world_frame, tf2buf=self.tfbuffer)

        # Objects found proto
        # If the last action is "find", and we receive object detections
        # that contain target objects, then these objects will be considered 'found'
        if isinstance(self.last_action, a_pb2.Find):
            for det_pb in detections_pb.detections:
                if det_pb.label in self.agent_config["targets"]:
                    self.objects_found.add(det_pb.label)
        header = proto_utils.make_header(frame_id=self.world_frame)
        objects_found_pb = o_pb2.ObjectsFound(
            header=header, robot_id=self.robot_id,
            object_ids=sorted(list(self.objects_found)))

        # Robot pose proto
        robot_pose_tuple = ros2_utils.pose_to_tuple(robot_pose_msg.pose)
        robot_pose_pb = o_pb2.RobotPose(
            header=header,
            robot_id=self.robot_id,
            pose_3d=proto_utils.posetuple_to_poseproto(robot_pose_tuple))
        return detections_pb, robot_pose_pb, objects_found_pb

    def wait_for_robot_pose(self):
        robot_pose_msg = ros2_utils.wait_for_messages(
            [self._robot_pose_topic], [geometry_msgs.PoseStamped],
            verbose=True)[0]
        robot_pose_tuple = ros2_utils.pose_to_tuple(robot_pose_msg.pose)
        return robot_pose_tuple

    def make_nav_action(self, pos, orien, action_id, nav_type):
        if nav_type not in {"2d", "3d"}:
            raise ValueError("nav_type should be '2d' or '3d'")
        goal_keys = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
        goal_values = [*pos, *orien]
        nav_action = KeyValAction(stamp=self.get_clock().now().to_msg(),
                                  type="nav",
                                  keys=["action_id"] + goal_keys + ["nav_type"],
                                  values=list(map(str, [action_id] + goal_values + [nav_type])))
        return nav_action

    def execute_action(self, action_id, action_pb):
        """All viewpoint movement actions specify a goal pose
        the robot should move its end-effector to, and publish
        that as a KeyValAction."""
        if isinstance(action_pb, a_pb2.MoveViewpoint):
            if action_pb.HasField("dest_3d"):
                dest = proto_utils.poseproto_to_posetuple(action_pb.dest_3d)
                nav_type = "3d"

            elif action_pb.HasField("dest_2d"):
                robot_pose = np.asarray(self.wait_for_robot_pose())
                dest_2d = proto_utils.poseproto_to_posetuple(action_pb.dest_2d)
                x, y, thz = dest_2d
                z = robot_pose[2]
                thx, thy, _ = math_utils.quat_to_euler(*robot_pose[3:])
                dest = (x, y, z, *math_utils.euler_to_quat(thx, thy, thz))
                nav_type = "2d"
                self.clear_octree_markers()  # 2d action means there's no 3D belief.

            else:
                raise NotImplementedError("Not implemented action_pb.")

            nav_action = self.make_nav_action(
                dest[:3], dest[3:], action_id, nav_type)
            self._action_pub.publish(nav_action)
            self.get_logger().info("published nav action for execution")

        elif isinstance(action_pb, a_pb2.Find):
            find_action = KeyValAction(stamp=self.get_clock().now().to_msg(),
                                       type="find",
                                       keys=["action_id"],
                                       values=[action_id])
            self._action_pub.publish(find_action)
            self.get_logger().info("published find action for execution")

    @property
    def ros_visual_config(self):
        return self.agent_config.get("misc", {}).get("ros_visual", {})


    def plan_action(self):
        response_plan = self._genmos_client.planAction(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        action_pb = proto_utils.interpret_planned_action(response_plan)
        action_id = response_plan.action_id
        self.get_logger().info("plan action finished. Action ID: {}".format(typ.info(action_id)))
        self.last_action = action_pb
        return action_id, action_pb

    def wait_observation_and_update_belief(self, action_id):
        # Now, wait for observation, and then update belief
        detections_pb, robot_pose_pb, objects_found_pb =\
            self.wait_for_observation()
        # send obseravtions for belief update
        header = proto_utils.make_header(frame_id=self.world_frame)
        response_observation = self._genmos_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=detections_pb,
            objects_found=objects_found_pb,
            header=header, return_fov=True,
            action_id=action_id, action_finished=True, debug=False)
        response_robot_belief = self._genmos_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        return response_observation, response_robot_belief, detections_pb

    def run(self):
        # First, create an agent
        self._genmos_client.createAgent(
            header=proto_utils.make_header(), config=self.agent_config,
            robot_id=self.robot_id)

        # Make the client listen to server
        ls_future = self._genmos_client.listenToServer(
            self.robot_id, self.server_message_callback)
        self._local_robot_id = None  # needed if the planner is hierarchical

        # Update search region
        self.update_search_region()

        # wait for agent creation
        self.get_logger().info("waiting for genmos agent creation...")
        self._genmos_client.waitForAgentCreation(self.robot_id)
        self.get_logger().info("agent created!")

        # visualize initial belief
        self.get_and_visualize_belief()
        self.get_logger().info("belief visualized!")

        # create planner
        response = self._genmos_client.createPlanner(config=self.planner_config,
                                                    header=proto_utils.make_header(),
                                                    robot_id=self.robot_id)
        self.get_logger().info("planner created!")

        # Send planning requests
        for step in range(self.config["task_config"]["max_steps"]):
            self.get_logger().info(typ.cyan(f"Step {step}"))
            action_id, action_pb = self.plan_action()
            self.clear_fovs_markers()  # clear fovs markers before executing action
            self.execute_action(action_id, action_pb)
            ros2_utils.wait_for_messages(
                self, [self._action_done_topic], [std_msgs.String],
                allow_headerless=True, verbose=True,
                callback_group=self.wfm_cb_group)
            self.get_logger().info(typ.success("action done."))

            if self.dynamic_update:
                self.update_search_region()

            response_observation, response_robot_belief, detections_pb =\
                self.wait_observation_and_update_belief(action_id)
            self.get_logger().info(f"\n detections:\n {proto_utils.parse_detections_proto(detections_pb)}")
            robot_belief_pb = response_robot_belief.robot_belief
            objects_found = set(robot_belief_pb.objects_found.object_ids)
            self.objects_found.update(objects_found)
            self.get_logger().info(f"\nrobot belief:\n  pose: {robot_belief_pb.pose.pose_3d}"\
                                   f"\n  objects found: {objects_found}")
            # visualize FOV and belief
            self.get_and_visualize_belief()
            if response_observation.HasField("fovs"):
                self.visualize_fovs_3d(response_observation)

            # Check if we are done
            if objects_found == set(self.agent_config["targets"]):
                self.get_logger().info("Done!")
                break
            self.get_logger().info("--------------")
            time.sleep(1)
