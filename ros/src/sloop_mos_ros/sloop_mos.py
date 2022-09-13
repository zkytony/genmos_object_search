import rospy
import tf2_ros
import math
import numpy as np
import time
import pickle
import json
from pomdp_py.utils import typ

import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import vision_msgs.msg as vision_msgs
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
from sloop_object_search.utils.misc import import_class


SEARCH_SPACE_RESOLUTION_3D = 0.1
SEARCH_SPACE_RESOLUTION_2D = 0.3

def make_nav_action(pos, orien, action_id):
    goal_keys = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
    goal_values = [*pos, *orien]
    nav_action = KeyValAction(stamp=rospy.Time.now(),
                              type="nav",
                              keys=["action_id"] + goal_keys,
                              values=list(map(str, [action_id] + goal_values)))
    return nav_action

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
            region_size_x=search_region_config.get("region_size_x"),
            region_size_y=search_region_config.get("region_size_y"),
            region_size_z=search_region_config.get("region_size_z"),
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
                bobj_pb, header, alpha_scaling=1.0)
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
                header = std_msgs.Header(stamp=rospy.Time.now(),
                                         frame_id=self.world_frame)
                clear_msg = ros_utils.clear_markers(header, ns="")
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
        robot_pose_msg, object_detections_msg = ros_utils.WaitForMessages(
            [self._robot_pose_topic, self._object_detections_topic],
            [geometry_msgs.PoseStamped, vision_msgs.Detection3DArray],
            queue_size=self.obqueue_size, delay=self.obdelay, verbose=True).messages

        if robot_pose_msg.header.frame_id != self.world_frame:
            # Need to convert robot pose to world frame
            robot_pose_msg = ros_utils.tf2_transform(self.tfbuffer, robot_pose_msg, self.world_frame)

        # Detection proto
        detections_pb = ros_utils.detection3darray_to_proto(
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
        robot_pose_tuple = ros_utils.pose_to_tuple(robot_pose_msg.pose)
        robot_pose_pb = o_pb2.RobotPose(
            header=header,
            robot_id=self.robot_id,
            pose_3d=proto_utils.posetuple_to_poseproto(robot_pose_tuple))
        return detections_pb, robot_pose_pb, objects_found_pb

    def wait_for_robot_pose(self):
        robot_pose_msg = ros_utils.WaitForMessages(
            [self._robot_pose_topic], [geometry_msgs.PoseStamped],
            verbose=True).messages[0]
        robot_pose_tuple = ros_utils.pose_to_tuple(robot_pose_msg.pose)
        return robot_pose_tuple

    def execute_action(self, action_id, action_pb):
        """All viewpoint movement actions specify a goal pose
        the robot should move its end-effector to, and publish
        that as a KeyValAction."""
        if isinstance(action_pb, a_pb2.MoveViewpoint):
            if action_pb.HasField("dest_3d"):
                dest = proto_utils.poseproto_to_posetuple(action_pb.dest_3d)

            elif action_pb.HasField("dest_2d"):
                robot_pose = np.asarray(self.wait_for_robot_pose())
                dest_2d = proto_utils.poseproto_to_posetuple(action_pb.dest_2d)
                x, y, thz = dest_2d
                z = robot_pose[2]
                thx, thy, _ = math_utils.quat_to_euler(*robot_pose[3:])
                dest = (x, y, z, *math_utils.euler_to_quat(thx, thy, thz))
            else:
                raise NotImplementedError("Not implemented action_pb.")

            nav_action = make_nav_action(dest[:3], dest[3:], action_id)
            self._action_pub.publish(nav_action)
            rospy.loginfo("published nav action for execution")

        elif isinstance(action_pb, a_pb2.Find):
            find_action = KeyValAction(stamp=rospy.Time.now(),
                                       type="find",
                                       keys=["action_id"],
                                       values=[action_id])
            self._action_pub.publish(find_action)
            rospy.loginfo("published find action for execution")

    def setup(self):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node(self.name)

        # Initialize grpc client
        self._sloop_client = SloopObjectSearchClient()
        config = rospy.get_param("~config")  # access parameters together as a dictionary
        self.config = config
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

        # tf; need to create listener early enough before looking up to let tf propagate into buffer
        # reference: https://answers.ros.org/question/292096/right_arm_base_link-passed-to-lookuptransform-argument-target_frame-does-not-exist/
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)

        # Remap these topics that we subscribe to, if needed.
        self._search_region_2d_point_cloud_topic = "~search_region_cloud_2d"
        self._search_region_3d_point_cloud_topic = "~search_region_cloud_3d"
        self._robot_pose_topic = "~robot_pose"
        # Note for object detections, we accept 3D bounding-box-based detections.
        self._object_detections_topic = "~object_detections"
        self._detection_vision_info_topic = "~vision_info"
        self._action_done_topic = "~action_done"

        # additional parameters
        self.obqueue_size = rospy.get_param("~obs_queue_size", 200)
        self.obdelay = rospy.get_param("~obs_delay", 0.5)

        # Need to wait for vision info
        vinfo_msg = ros_utils.WaitForMessages([self._detection_vision_info_topic],
                                              [vision_msgs.VisionInfo], verbose=True).messages[0]
        self.detection_class_names = rospy.get_param("/" + vinfo_msg.database_location)

        # Planning-related
        self.last_action = None
        self.objects_found = set()


    def run(self):
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

        # Send planning requests
        for step in range(self.config["task_config"]["max_steps"]):
            response_plan = self._sloop_client.planAction(
                self.robot_id, header=proto_utils.make_header(self.world_frame))
            action_pb = proto_utils.interpret_planned_action(response_plan)
            action_id = response_plan.action_id
            rospy.loginfo("plan action finished. Action ID: {}".format(typ.info(action_id)))
            self.last_action = action_pb

            self.execute_action(action_id, action_pb)
            ros_utils.WaitForMessages([self._action_done_topic], [std_msgs.String],
                                      allow_headerless=True, verbose=True)
            rospy.loginfo(typ.success("action done."))

            # Now, wait for observation, and then update belief
            detections_pb, robot_pose_pb, objects_found_pb = self.wait_for_observation()
            # send obseravtions for belief update
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
            self.get_and_visualize_belief()
            if response_observation.HasField("fovs"):
                self.visualize_fovs_3d(response_observation)

            # Check if we are done
            if objects_found == set(self.agent_config["targets"]):
                rospy.loginfo("Done!")
                break
            time.sleep(1)
