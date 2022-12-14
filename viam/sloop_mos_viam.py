# Currently, this script contains mocked data specific for the
# Viam Lab setup and integrates with ROS RViZ for visaulization. It
# is a TODO to make this more generalized to different robots, but
# the issue is that right now viam_utils functions are only tested
# for this Viam Lab setup, and there is ROS/RViZ integration which is
# not what Viam wants I suppose.  However, this provides a SloopMosViam
# class that can be the basis of that more general program.
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
import json
import numpy as np
from pomdp_py.utils import typ

# Allow importing stuff from parent folder
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import constants

# Viam related
from utils import viam_utils
import viam.proto.common as v_pb2

# ROS related
import rospy
import ros_numpy
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as viz_msgs
import sensor_msgs.msg as sensor_msgs
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
from sloop_object_search.utils.misc import import_class, hash16


WORKING_MOTION_POSES = {
    (-0.6, -0.4, 0.06, *viam_utils.ovec_to_quat(0, -1, 0, 90)),
    (-0.6, -0.4, 1.60, *viam_utils.ovec_to_quat(0, -1, 0, 90)),
    (-0.6, -0.4, 0.60, *viam_utils.ovec_to_quat(0, -1, 0, 90)),
    (-0.7, -0.4, 0.60, *viam_utils.ovec_to_quat(0, -1, 0, 90)),
    (-0.5, -1.3, 0.53, *viam_utils.ovec_to_quat(0, -1, 0, 90)),
    (-0.42, -1.3, 0.53, *viam_utils.ovec_to_quat(0, -1, 0, 90))
}


class SloopMosViam:
    def __init__(self, name="sloop_viam"):
        self.name = name
        self.sloop_client = None  # connection to sloop server
        self.viam_robot = None  # connection to the viam robot

    def setup(self, viam_robot, viam_names, config, world_frame):
        """
        Args:
            viam_robot: the grpc connection to viam
            viam_names (dict): maps from a string (e.g. 'color_camera')
                to another string (e.g. 'gripper-main:color-cam').
            config (dict): misc configurations
            world_frame (str): name of the world frame.

        Note:
           required entries in viam_names:
           - color_camera
           - depth_camera
           - detector
           - arm TODO: right now the code here is specific to robot arm. Make it general.
        """
        self.setup_for_rviz()

        self.viam_robot = viam_robot
        self.viam_names = viam_names

        # Configuration and parameters
        self.config = config
        self.agent_config = config["agent_config"]
        self.planner_config = config["planner_config"]
        self.robot_id = self.agent_config["robot"]["id"]
        self.world_state_config = config["world_state_config"]
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
        self._world_state_cloud_pub = rospy.Publisher(
            "~world_state_cloud", sensor_msgs.PointCloud2, queue_size=10, latch=True)
        self._goal_viz_pub = rospy.Publisher(
            "~goal_markers", viz_msgs.MarkerArray, queue_size=10, latch=True)

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
        geoms_F_world = []  # geometry with pose in specified reference frame.
        viz_markers = []
        for obj_spec in self.world_state_config:
            pose = obj_spec["pose"]  # in meters; center of the box
            sizes = obj_spec["sizes"]  # in meters
            geom = v_pb2.Geometry(center=v_pb2.Pose(x=pose[0]*1000,
                                                    y=pose[1]*1000,
                                                    z=pose[2]*1000),
                                  box=v_pb2.RectangularPrism(
                                      dims_mm=v_pb2.Vector3(
                                          x=sizes[0]*1000,
                                          y=sizes[1]*1000,
                                          z=sizes[2]*1000)))

            # think of this as ROS's PoseStamped (pose with frame)
            geom_F_world = v_pb2.GeometriesInFrame(
                reference_frame=self.world_frame, geometries=[geom])
            geoms_F_world.append(geom_F_world)

            viz_marker = ros_utils.make_viz_marker_for_object(
                obj_spec["name"], pose,
                std_msgs.Header(stamp=rospy.Time.now(), frame_id=self.world_frame),
                viz_type=viz_msgs.Marker.CUBE,
                color=obj_spec["color"],
                scale=geometry_msgs.Vector3(x=sizes[0],
                                            y=sizes[1],
                                            z=sizes[2]),
                lifetime=0)
            viz_markers.append(viz_marker)

            # record pose for periodic tf publication
            self._world_frame_poses[obj_spec["name"]] = pose[:3]

        world_state = v_pb2.WorldState(obstacles=geoms_F_world)
        self.viam_world_state = world_state

        self._world_state_markers_pub.publish(viz_msgs.MarkerArray(viz_markers))
        rospy.loginfo("Published world state markers")

        rospy.loginfo("Publishing world state TF")
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
                m = ros_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=hit_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            break  # just visualize one
        self._fovs_markers_pub.publish(viz_msgs.MarkerArray(markers))

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
        # if a 3D box for sample space is specified, visualize that too
        if "sample_space" in self.agent_config["robot"]["action"]["topo"]:
            sample_space = self.agent_config["robot"]["action"]["topo"]["sample_space"]
            center = sample_space["center_x"], sample_space["center_y"], sample_space["center_z"]
            sizes = sample_space["size_x"], sample_space["size_y"], sample_space["size_z"]
            msg = ros_utils.make_viz_marker_cube(
                hash16(self.robot_id + "_topo_map_sample_space"),
                (*center, 0., 0., 0., 1.),
                header, color=[0.05, 0.4, 0.48, 0.2],
                scale=geometry_msgs.Vector3(x=sizes[0],
                                            y=sizes[1],
                                            z=sizes[2]),
                lifetime=0)
            markers.append(msg)
            rospy.loginfo("adding marker for topo map sample space")
        self._topo_map_3d_markers_pub.publish(viz_msgs.MarkerArray(markers))
        rospy.loginfo("topo map 3d visualized")

        robot_pose = proto_utils.robot_pose_from_proto(robot_belief_pb.pose)
        robot_marker, trobot = ros_utils.viz_msgs_for_robot_pose(
            robot_pose, self.world_frame, self.robot_id,
            color=[0.9, 0.1, 0.1, 0.9], lifetime=0,
            scale=geometry_msgs.Vector3(x=0.6, y=0.08, z=0.08))
        self._robot_pose_markers_pub.publish(viz_msgs.MarkerArray([robot_marker]))
        self._world_frame_poses[self.robot_id] = robot_pose
        self.tf2br.sendTransform(trobot)

    def _simulate_point_cloud_from_world_state(self, density=0.01, viz=False):
        """
        Returns a numpy array (Nx3) of point cloud generated
        based on geometrical information in 'self.world_state_config'.

        density (int): the number of points per cubic meter (1m^3).
        viz (bool): True if publishes the world state point cloud for RVIZ.
        """
        # currently, all world state objects are boxes.
        sim_cloud_arr = None
        for obj_spec in self.world_state_config:
            if obj_spec.get("skip_cloud", False):
                # do not include this object when building the point cloud
                continue

            # note that density is cm^3 while sizes are in meters
            # need to convert the volume to cm^3 as well
            volume = (obj_spec["sizes"][0] * obj_spec["sizes"][1] * obj_spec["sizes"][2]) * 100**3
            num_points = int(volume * density)

            x_range = (obj_spec["pose"][0] - obj_spec["sizes"][0] / 2.,
                       obj_spec["pose"][0] + obj_spec["sizes"][0] / 2.)
            y_range = (obj_spec["pose"][1] - obj_spec["sizes"][1] / 2.,
                       obj_spec["pose"][1] + obj_spec["sizes"][1] / 2.)
            z_range = (obj_spec["pose"][2] - obj_spec["sizes"][2] / 2.,
                       obj_spec["pose"][2] + obj_spec["sizes"][2] / 2.)
            obj_points = np.random.uniform(low=[x_range[0], y_range[0], z_range[0]],
                                           high=[x_range[1], y_range[1], z_range[1]],
                                           size=[num_points, 3])
            if sim_cloud_arr is None:
                sim_cloud_arr = obj_points
            else:
                sim_cloud_arr = np.append(sim_cloud_arr, obj_points, axis=0)

        if viz:
            # publish and latch the point cloud
            point_cloud_msg = ros_utils.xyz_array_to_pointcloud2(
                sim_cloud_arr, frame_id=self.world_frame)
            self._world_state_cloud_pub.publish(point_cloud_msg)
            print("Published point cloud!")

        return sim_cloud_arr

    @property
    def search_space_res_3d(self):
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        return search_region_config.get("res", constants.SEARCH_SPACE_RESOLUTION_3D)

    def update_search_region_3d(self, init=False):
        """
        Sends gRPC request to update search region based on point cloud
        observation. If 'init' is True, then this function is called
        to initialize the search agent in the server. Otherwise, this
        function is called during search.
        """
        print("Sending request to update search region (3D)")

        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        point_cloud_from_world_state = search_region_config.get("point_cloud_from_world_state", False)
        if point_cloud_from_world_state:
            # will create a synthetic point cloud based on the world state
            # specified in the config
            cloud_arr = self._simulate_point_cloud_from_world_state(viz=True)
        else:
            # will try to obtain point cloud from camera
            try:
                cloud_arr = viam_utils.viam_get_point_cloud_array(
                    self.viam_robot, target_frame=self.world_frame)
            except Exception:
                print("Failed to obtain point cloud. Will proceed with empty point cloud.")
                cloud_arr = np.array([])

        cloud_pb = proto_utils.pointcloudproto_from_array(cloud_arr, self.world_frame)

        # parameters
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

            # visualize the goal
            _marker_scale = geometry_msgs.Vector3(x=0.4, y=0.05, z=0.05)
            _goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
                self.robot_id + "_goal", dest, std_msgs.Header(frame_id=self.world_frame, stamp=rospy.Time.now()),
                scale=_marker_scale, color=[0.53, 0.95, 0.99, 0.9], lifetime=0)
            self._goal_viz_pub.publish(viz_msgs.MarkerArray([_goal_marker_msg]))

            # execute the goal
            print("Executing nav action (viewpoint movement)")
            dest_viam = self._output_viam_pose(dest)
            # WARN: instead of doing going to the destination, go to
            # a closest pose known to work.
            approx_dest_viam = min(WORKING_MOTION_POSES, key=lambda p: math_utils.euclidean_dist(p[:3], dest_viam[:3]))
            success = await viam_utils.viam_move(self.viam_robot,
                                                 self.viam_names["arm"],
                                                 approx_dest_viam, self.world_frame,
                                                 self.viam_world_state)
            if not success:
                print("viewpoint movement failed.")
            else:
                print("viewpoint movement succeeded.")

        elif isinstance(action_pb, a_pb2.Find):
            print("Signaling find action")
            # TODO: signal find
            success = await viam_utils.viam_signal_find(self.viam_robot)
            if success:
                print("Find action taken.")

    def _process_viam_pose(self, viam_pose):
        """
        Args:
            viam_pose (x,y,z,qx,qy,qz,qw): pose returned wrt Viam's frame system (y forward)
        Returns:
            pose: pose wrt conventional frame system (x forward)
        """
        # get transform to align 0 quat direction from viam's system (+z) to my system (+x)
        fixed_transform = math_utils.R_euler(0, -90, 0, affine=True)
        vx, vy, vz, vqx, vqy, vqz, vqw = viam_pose
        pose_transform = np.matmul(math_utils.T(vx, vy, vz),
                                   np.matmul(math_utils.R_quat(vqx, vqy, vqz, vqw, affine=True),
                                             fixed_transform))
        # apply fixed transform, then extract the pose
        pose_quat = math_utils.R_to_quat(math_utils.R_matrix(pose_transform[:3, :3]))
        pose_xyz = pose_transform[:3, 3]
        return (*pose_xyz, *pose_quat)

    def _output_viam_pose(self, pose):
        """given a pose in my frame system (+x), output a pose in Viam's
        frame system (+z). """
        # get transform to align 0 quat direction from my system (+x) to viam's system (+z)
        fixed_transform = math_utils.R_euler(0, 90, 0, affine=True)
        x, y, z, qx, qy, qz, qw = pose
        viam_pose_transform = np.matmul(math_utils.T(x, y, z),
                                        np.matmul(math_utils.R_quat(qx, qy, qz, qw, affine=True),
                                                  fixed_transform))
        # apply fixed transform, then extract the pose
        viam_pose_quat = math_utils.R_to_quat(math_utils.R_matrix(viam_pose_transform[:3, :3]))
        viam_pose_xyz = viam_pose_transform[:3, 3]
        return (*viam_pose_xyz, *viam_pose_quat)


    async def wait_for_observation(self):
        """We wait for the robot pose (PoseStamped) and the
        object detections (vision_msgs.Detection3DArray)

        Returns:
            a tuple: (detections_pb, robot_pose_pb, objects_found_pb)"""
        # TODO: future viam: time sync between robot pose and object detection
        robot_pose_viam = await viam_utils.viam_get_ee_pose(self.viam_robot, arm_name=self.viam_names["arm"])
        robot_pose = self._process_viam_pose(robot_pose_viam)
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose_viam)
        # Note: right now we only get 2D detection
        detections = await viam_utils.viam_get_object_detections2d(
            self.viam_robot,
            camera_name=self.viam_names["color_camera"],
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

    async def stream_state(self):
        search_region_config = self.agent_config.get("search_region", {}).get("3d", {})
        point_cloud_from_world_state = search_region_config.get("point_cloud_from_world_state", False)
        if point_cloud_from_world_state:
            # will create a synthetic point cloud based on the world state
            # specified in the config
            cloud_arr = self._simulate_point_cloud_from_world_state(viz=True)

        self.publish_world_state_in_ros()

        # TODO: future viam: time sync between robot pose and object detection
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            # visualize the robot state
            header = std_msgs.Header(stamp=rospy.Time.now(),
                                     frame_id=self.world_frame)
            clear_msg = ros_utils.clear_markers(header, ns="")
            self._robot_pose_markers_pub.publish(clear_msg)

            robot_pose_viam = await viam_utils.viam_get_ee_pose(self.viam_robot, arm_name=self.viam_names["arm"])
            print("got robot pose from viam", robot_pose_viam)
            robot_pose = self._process_viam_pose(robot_pose_viam)
            robot_marker, trobot = ros_utils.viz_msgs_for_robot_pose(
                robot_pose, self.world_frame, self.robot_id,
                color=[0.9, 0.1, 0.1, 0.9], lifetime=0,
                scale=geometry_msgs.Vector3(x=0.6, y=0.08, z=0.08))
            self._robot_pose_markers_pub.publish(viz_msgs.MarkerArray([robot_marker]))
            self._world_frame_poses[self.robot_id] = robot_pose
            self.tf2br.sendTransform(trobot)
            rate.sleep()


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
        self.update_search_region_3d(init=True)

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
            self.get_and_visualize_belief_3d()
            if response_observation.HasField("fovs"):
                self.visualize_fovs_3d(response_observation)

            # Check if we are done
            if objects_found == set(self.agent_config["targets"]):
                rospy.loginfo("Done!")
                break
            time.sleep(1)
