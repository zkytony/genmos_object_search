# Code written specifically for the test at Viam Lab on the UR5 robot
# TODO: Currently, this script contains mocked data specific for the
# Viam Lab setup and integrates with ROS RViZ for visaulization. It
# is a TODO to make this more generalized to different robots, but
# the issue is that right now viam_utils functions are only tested
# for this Viam Lab setup, and there is ROS/RViZ integration which is
# not what Viam wants I suppose.  However, this provides a SloopMosViam
# class that can be the basis of that more general program.
#
# Viam Robot Pose
# Connected!
# (0.2797589770640316, 0.7128048233719448, 0.5942370817926967, -0.6500191634979094, 0.4769735333791088, 0.4926158987104014, 0.32756817897816304)
# position {
#   x: 0.27975897706403158
#   y: 0.71280482337194484
#   z: 0.59423708179269674
# }
# rotation {
#   x: -0.65001916349790945
#   y: 0.4769735333791088
#   z: 0.4926158987104014
#   w: 0.32756817897816304
# }
#
# Example output of object detection
# header {
# }
# robot_id: "robot0"
# detections {
#   label: "Chair"
#   box_2d {
#     x_min: 69
#     y_min: 146
#     x_max: 161
#     y_max: 237
#   }
# }
# detections {
#   label: "Person"
#   box_2d {
#     x_min: 14
#     y_min: 87
#     x_max: 38
#     y_max: 127
#   }
# }
# detections {
#   label: "Person"
#   box_2d {
#     x_min: 195
#     y_min: 31
#     x_max: 226
#     y_max: 104
#   }
# }
##################################
#
# 1. run in one terminal, run 'python -m sloop_object_search.grpc.server'
# 2. run in one terminal, run 'python test_run_search_viamlab_ur5.py'
# 3. run in one terminal, run 'roslaunch view_viam_search.launch'
import asyncio
import yaml
import os
import sys
import time
import numpy as np
from pomdp_py.utils import typ

# Allow importing stuff from parent folder
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import constants

# Viam related
import viam_utils
import viam.proto.common as v_pb2

# ROS related
import rospy
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as viz_msgs
from sloop_mos_ros import ros_utils
from tf2_ros import TransformBroadcaster

from sloop_object_search.grpc.client import SloopObjectSearchClient
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc import sloop_object_search_pb2 as slpb2
from sloop_object_search.grpc import observation_pb2 as o_pb2
from sloop_object_search.grpc import action_pb2 as a_pb2
from sloop_object_search.grpc import common_pb2
from sloop_object_search.grpc.common_pb2 import Status
from sloop_object_search.grpc.constants import Message
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils.misc import import_class


class SloopMosViam:
    def __init__(self, name="sloop_viam"):
        self.name = name
        self.sloop_client = None  # connection to sloop server
        self.viam_robot = None  # connection to the viam robot

    def setup(self, viam_robot, viam_names, config, world_frame):
        self.setup_for_rviz()

        self.viam_robot = viam_robot
        self.viam_names = viam_names

        # Configuration and parameters
        self.config = config
        self.agent_config = config["agent_config"]
        self.planner_config = config["planner_config"]
        self.robot_id = self.agent_config["robot"]["id"]
        self.world_frame = world_frame  # fixed frame of the world

        # Initialize grpc client
        self.sloop_client = SloopObjectSearchClient()

        # Planning-related
        self.last_action = None
        self.objects_found = set()


    def setup_for_rviz(self):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node(self.name)

        # Initialize visualization marker publishers
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", viz_msgs.MarkerArray, queue_size=10, latch=True)
        self._fovs_markers_pub = rospy.Publisher(
            "~fovs", viz_msgs.MarkerArray, queue_size=10, latch=True)
        self._topo_map_3d_markers_pub = rospy.Publisher(
            "~topo_map_3d", viz_msgs.MarkerArray, queue_size=10, latch=True)
        self._robot_pose_markers_pub = rospy.Publisher(
            "~robot_pose_marker", viz_msgs.MarkerArray, queue_size=10, latch=True)

        # world state
        self._world_state_markers_pub = rospy.Publisher(
            "~world_state", viz_msgs.MarkerArray, queue_size=10, latch=True)

        # TF broadcaster
        self.tf2br = TransformBroadcaster()
        self._world_frame_poses = {}  # maps from frame name to the pose
                                      # (position or pose) w.r.t. world frame
                                      # that should be published as tf transforms


    def publish_world_state_in_ros(self):
        """Publishes the TF and visualization markers for the world state
        periodically"""
        ## Hard coded for the Viam Lab scene
        table_pose = (0.43, 0, -0.49)  # center of table in world frame
        table_sizes = (0.86, 1.16, 0.93)

        xarm_pose = (0.8, 0., 0.475) # in world frame
        xarm_sizes = (0.4, 0.4, 1.0)

        chair_pose = (0.05, 1.37, -0.54) # in world frame
        chair_sizes = (0.4, 0.4, 0.83)

        arm_origin_pose = (0.0, 0.0, 0.0) # in world frame
        arm_origin_sizes = (0.17, 0.17, 0.25)

        table = v_pb2.Geometry(center=v_pb2.Pose(x=table_pose[0]*1000,
                                                 y=table_pose[1]*1000,
                                                 z=table_pose[2]*1000),
                               box=v_pb2.RectangularPrism(
                                   dims_mm=v_pb2.Vector3(
                                       x=table_sizes[0]*1000,
                                       y=table_sizes[1]*1000,
                                       z=table_sizes[2]*1000)))
        tableFrame = v_pb2.GeometriesInFrame(
            reference_frame=self.world_frame, geometries=[table])
        table_marker = ros_utils.make_viz_marker_for_object(
            "table", table_pose,
            std_msgs.Header(stamp=rospy.Time.now(), frame_id=self.world_frame),
            viz_type=viz_msgs.Marker.CUBE,
            color=[0.05, 0.45, 0.72, 0.8],
            scale=geometry_msgs.Vector3(x=table_sizes[0],
                                        y=table_sizes[1],
                                        z=table_sizes[2]),
            lifetime=0)  # 0 is forever

        xARM = v_pb2.Geometry(center=v_pb2.Pose(x=xarm_pose[0]*1000,
                                                y=xarm_pose[1]*1000,
                                                z=xarm_pose[2]*1000),
                              box=v_pb2.RectangularPrism(
                                  dims_mm=v_pb2.Vector3(
                                      x=xarm_sizes[0]*1000,
                                      y=xarm_sizes[1]*1000,
                                      z=xarm_sizes[2]*1000)))
        xARMFrame = v_pb2.GeometriesInFrame(reference_frame=self.world_frame, geometries=[xARM])
        xarm_marker = ros_utils.make_viz_marker_for_object(
            "xarm", xarm_pose,
            std_msgs.Header(stamp=rospy.Time.now(), frame_id=self.world_frame),
            viz_type=viz_msgs.Marker.CUBE,
            color=[0.95, 0.95, 0.88, 0.8],
            scale=geometry_msgs.Vector3(x=xarm_sizes[0],
                                        y=xarm_sizes[1],
                                        z=xarm_sizes[2]),
            lifetime=0)  # 0 is forever

        world_state = v_pb2.WorldState(obstacles=[tableFrame, xARMFrame])
        self.viam_world_state = world_state

        # Chair
        chair_marker = ros_utils.make_viz_marker_for_object(
            "chair", chair_pose,
            std_msgs.Header(stamp=rospy.Time.now(), frame_id=self.world_frame),
            viz_type=viz_msgs.Marker.CUBE,
            color=[0.18, 0.18, 0.18, 0.8],
            scale=geometry_msgs.Vector3(x=chair_sizes[0],
                                        y=chair_sizes[1],
                                        z=chair_sizes[2]),
            lifetime=0)  # 0 is forever

        # Robot arm base
        arm_origin_marker = ros_utils.make_viz_marker_for_object(
            "arm_origin", arm_origin_pose,
            std_msgs.Header(stamp=rospy.Time.now(), frame_id=self.world_frame),
            viz_type=viz_msgs.Marker.CUBE,
            color=[0.5, 0.8, 0.9, 0.8],
            scale=geometry_msgs.Vector3(x=arm_origin_sizes[0],
                                        y=arm_origin_sizes[1],
                                        z=arm_origin_sizes[2]),
            lifetime=0)  # 0 is forever

        markers = [table_marker, xarm_marker, chair_marker, arm_origin_marker]
        self._world_state_markers_pub.publish(viz_msgs.MarkerArray(markers))
        rospy.loginfo("Published world state markers")

        # periodically publish tf
        self._world_frame_poses["table"] = table_pose[:3]
        self._world_frame_poses["xarm"] = xarm_pose[:3]
        self._world_frame_poses["chair"] = xarm_pose[:3]
        self._world_frame_poses["arm_base"] = xarm_pose[:3]

        rospy.Timer(rospy.Duration(1./5),
                    lambda event: self._publish_tf())

    def _publish_tf(self):
        for child_frame in self._world_frame_poses:
            pose = self._world_frame_poses[child_frame]

            if len(pose) == 3:
                tf_msg = ros_utils.tf2msg_from_object_loc(
                    pose[:3], self.world_frame, child_frame)
            else:
                _, tf_msg = ros_utils.viz_msgs_for_robot_pose(
                    pose, self.world_frame, child_frame)
            self.tf2br.sendTransform(tf_msg)

    def get_and_visualize_belief_3d(self):
        robot_id = self.robot_id

        # Clear markers
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._octbelief_markers_pub.publish(clear_msg)
        self._topo_map_3d_markers_pub.publish(clear_msg)
        self._robot_pose_markers_pub.publish(clear_msg)

        response = self.sloop_client.getObjectBeliefs(
            robot_id, header=proto_utils.make_header(self.world_frame))
        if response.status != Status.SUCCESSFUL:
            print("Failed to get 3D belief")
            return
        rospy.loginfo("got belief")

        # visualize the belief
        header = std_msgs.Header(stamp=rospy.Time.now(),
                                 frame_id=self.world_frame)
        markers = []
        # First, visualize the belief of detected objects
        for bobj_pb in response.object_beliefs:
            if bobj_pb.object_id in self.objects_found:
                msg = ros_utils.make_octree_belief_proto_markers_msg(
                    bobj_pb, header, alpha_scaling=2.0, prob_thres=0.5)
                markers.extend(msg.markers)
        # For the other objects, just visualize one is enough.
        for bobj_pb in response.object_beliefs:
            if bobj_pb.object_id not in self.objects_found:
                msg = ros_utils.make_octree_belief_proto_markers_msg(
                    bobj_pb, header, alpha_scaling=1.0)
                markers.extend(msg.markers)
                break
        self._octbelief_markers_pub.publish(viz_msgs.MarkerArray(markers))

        rospy.loginfo("belief visualized")

        # visualize topo map in robot belief
        markers = []
        response_robot_belief = self.sloop_client.getRobotBelief(
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
        self._topo_map_3d_markers_pub.publish(viz_msgs.MarkerArray(markers))
        rospy.loginfo("topo map 3d visualized")

        # visualize robot pose vector
        robot_pose = proto_utils.robot_pose_from_proto(robot_belief_pb.pose)
        robot_marker, trobot = ros_utils.viz_msgs_for_robot_pose(
            robot_pose, self.world_frame, self.robot_id,
            color=[0.9, 0.1, 0.1, 0.9], lifetime=0,
            scale=geometry_msgs.Vector3(x=0.6, y=0.08, z=0.08))
        self._robot_pose_markers_pub.publish(viz_msgs.MarkerArray([robot_marker]))
        self._world_frame_poses[self.robot_id] = robot_pose
        self.tf2br.sendTransform(trobot)

    @property
    def search_space_res_3d(self):
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        return search_region_config.get("res", constants.SEARCH_SPACE_RESOLUTION_3D)


    def update_search_region_3d(self):
        print("Sending request to update search region (3D)")
        if viam_utils.MOCK:
            cloud_arr = np.array([])
        else:
            try:
                cloud_arr = viam_utils.viam_get_point_cloud_array(
                    self.viam_robot, target_frame=self.world_frame)
            except Exception:
                print("Failed to obtain point cloud. Will proceed with empty point cloud.")
                cloud_arr = np.array([])

        cloud_pb = proto_utils.pointcloudproto_from_array(cloud_arr, self.world_frame)

        # parameters
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        search_region_params_3d = dict(
            octree_size=search_region_config.get("octree_size", 32),
            search_space_resolution=search_region_config.get("res", constants.SEARCH_SPACE_RESOLUTION_3D),
            region_size_x=search_region_config.get("region_size_x"),
            region_size_y=search_region_config.get("region_size_y"),
            region_size_z=search_region_config.get("region_size_z"),
            debug=search_region_config.get("debug", False)
        )
        search_region_center = (search_region_config.get("center_x"),
                                search_region_config.get("center_y"),
                                search_region_config.get("center_z"), 0, 0, 0, 1)
        pose_pb = proto_utils.robot_pose_proto_from_tuple(search_region_center)
        self.sloop_client.updateSearchRegion(
            header=cloud_pb.header,
            robot_id=self.robot_id,
            robot_pose=pose_pb,
            point_cloud=cloud_pb,
            search_region_params_3d=search_region_params_3d)

    def plan_action(self):
        response_plan = self.sloop_client.planAction(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        action_pb = proto_utils.interpret_planned_action(response_plan)
        action_id = response_plan.action_id
        rospy.loginfo("plan action finished. Action ID: {}".format(typ.info(action_id)))
        self.last_action = action_pb
        return action_id, action_pb

    async def execute_action(self, action_id, action_pb):
        """All viewpoint movement actions specify a goal pose
        the robot should move its end-effector to, and publish
        that as a KeyValAction."""
        if isinstance(action_pb, a_pb2.MoveViewpoint):
            if action_pb.HasField("dest_3d"):
                dest = proto_utils.poseproto_to_posetuple(action_pb.dest_3d)
                nav_type = "3d"
            elif action_pb.HasField("dest_2d"):
                raise NotImplementedError("Not expecting destination to be 2D")
            else:
                raise NotImplementedError("Not implemented action_pb.")

            # TODO: action execution
            print("Executing nav action (viewpoint movement)")
            success = await viam_utils.viam_move(self.viam_robot,
                                                 self.viam_names["arm"],
                                                 dest, self.world_frame,
                                                 self.viam_world_state)
            if not success:
                print("viewpoint movement failed.")
            else:
                print("viewpoint movement succeeded.")

        elif isinstance(action_pb, a_pb2.Find):
            print("Signaling find action")
            # TODO: signal find
            success = viam_utils.viam_signal_find(self.viam_robot)
            if success:
                print("Find action taken.")

    async def wait_for_observation(self):
        """We wait for the robot pose (PoseStamped) and the
        object detections (vision_msgs.Detection3DArray)

        Returns:
            a tuple: (detections_pb, robot_pose_pb, objects_found_pb)"""
        # TODO: future viam: time sync between robot pose and object detection
        robot_pose = await viam_utils.viam_get_ee_pose(self.viam_robot, arm_name=self.viam_names["arm"])
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
        # Note: right now we only get 2D detection
        detections = await viam_utils.viam_get_object_detections2d(
            self.world_frame,
            camera_name=self.viam_names["camera"],
            detector_name=self.viam_names["detector"],
            confidence_thres=constants.DETECTION2D_CONFIDENCE_THRES)

        # Detection proto
        detections_pb = viam_utils.viam_detections2d_to_proto(self.robot_id, detections)

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
        robot_pose_pb = o_pb2.RobotPose(
            header=header,
            robot_id=self.robot_id,
            pose_3d=proto_utils.posetuple_to_poseproto(robot_pose))
        return detections_pb, robot_pose_pb, objects_found_pb

    async def wait_observation_and_update_belief(self, action_id):
        # Now, wait for observation, and then update belief
        detections_pb, robot_pose_pb, objects_found_pb = await self.wait_for_observation()
        # send obseravtions for belief update
        header = proto_utils.make_header(frame_id=self.world_frame)
        response_observation = self.sloop_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=detections_pb,
            objects_found=objects_found_pb,
            header=header, return_fov=True,
            action_id=action_id, action_finished=True, debug=False)
        response_robot_belief = self.sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        return response_observation, response_robot_belief

    def server_message_callback(self, message):
        print("received message:", message)
        raise NotImplementedError("Hierarchical planning is not yet integrated"\
                                  "for viam. Not expecting anything from the server.")

    async def run(self):
        # First, create an agent
        self.sloop_client.createAgent(
            header=proto_utils.make_header(), config=self.agent_config,
            robot_id=self.robot_id)

        # Make the client listen to server
        ls_future = self.sloop_client.listenToServer(
            self.robot_id, self.server_message_callback)
        self._local_robot_id = None  # needed if the planner is hierarchical

        # Update search region
        self.update_search_region_3d()

        # wait for agent creation
        print("waiting for sloop agent creation...")
        self.sloop_client.waitForAgentCreation(self.robot_id)
        print("agent created!")

        # visualize initial belief
        self.publish_world_state_in_ros()
        self.get_and_visualize_belief_3d()

        # create planner
        response = self.sloop_client.createPlanner(config=self.planner_config,
                                              header=proto_utils.make_header(),
                                              robot_id=self.robot_id)
        rospy.loginfo("planner created!")

        # Send planning requests
        for step in range(self.config["task_config"]["max_steps"]):
            action_id, action_pb = self.plan_action()
            await self.execute_action(action_id, action_pb)

            response_observation, response_robot_belief =\
                await self.wait_observation_and_update_belief(action_id)
            print(f"Step {step} robot belief:")
            robot_belief_pb = response_robot_belief.robot_belief
            objects_found = set(robot_belief_pb.objects_found.object_ids)
            objects_found.update(objects_found)
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



async def test_ur5e_viamlab(mock=False):
    with open("../config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)

    # Globally control whether we are using data from viam or mocked data
    viam_utils.MOCK = mock

    print(">>>>>>><<<<<<<<>>>> viam connecting >><<<<<<<<>>>>>>>")
    ur5robot = await viam_utils.connect_viamlab_ur5()
    viam_names = {
        "camera": "segmenter_cam",
        "detector": "find_objects",
        "arm": "arm"
    }
    world_frame = "arm_origin"

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    try:
        sloop_viam = SloopMosViam()
        sloop_viam.setup(ur5robot, viam_names, config, world_frame)
        await sloop_viam.run()
    finally:
        sloop_viam.sloop_client.channel.close()


if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab(mock=True))
