#!/usr/bin/env python
# To run this, use the 'simple_sim_env.launch' file:
#    roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=<map_name>
# Also run:
#    To get map point cloud, roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>
#    For rviz, roslaunch sloop_object_search_ros view_simple_sim.launch
# Then, you can run one of the test_* files.

import rospy
import pomdp_py
import math
import time

import actionlib
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import ColorRGBA, Header, String
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sloop_object_search_ros.msg import KeyValAction, KeyValObservation
from tf2_ros import TransformBroadcaster

from sloop_mos_ros import ros_utils
from sloop_mos_ros.framework import ActionExecutor
from sloop_object_search.oopomdp.domain.state import ObjectState, RobotState
from sloop_object_search.oopomdp.domain.action import MotionAction3D, FindAction
from sloop_object_search.oopomdp.domain.observation import ObjectVoxel, Voxel, ObjectDetection, GMOSObservation
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from sloop_object_search.oopomdp.models.observation_model import RobotObservationModel, GMOSObservationModel
from sloop_object_search.oopomdp.agent.common import (init_object_transition_models,
                                                      init_detection_models,
                                                      interpret_localization_model)
from sloop_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import quat_to_euler, euler_to_quat, to_rad, to_deg, euclidean_dist


class SimpleSimEnv(pomdp_py.Environment):
    """This is a simple 3D environment. All of its coordinates should
    be in the world frame, as this is meant to support the underlying
    simulation of an object search scenario in ROS."""
    def __init__(self, env_config):
        # Get initial robot pose
        self.env_config = env_config

        # agent config is the same format as what is given to a SLOOP MosAgent creation.
        self.agent_config = env_config["agent_config"]
        self._robot_pose_topic = "~init_robot_pose"
        robot_pose_msg = ros_utils.WaitForMessages([self._robot_pose_topic], [PoseStamped], verbose=True)\
                                  .messages[0]
        init_robot_pose = ros_utils.pose_tuple_from_pose_stamped(robot_pose_msg)

        self.robot_id = self.agent_config["robot"].get("id", "robot")
        init_robot_state = RobotState(self.robot_id,
                                      init_robot_pose, (), None)

        objects = self.agent_config["objects"]
        object_locations = self.env_config["object_locations"]
        object_states = {}
        for objid in objects:
            sobj = ObjectState(objid, objects[objid]["class"], tuple(object_locations[objid]))
            object_states[objid] = sobj
        init_state = pomdp_py.OOState({self.robot_id: init_robot_state,
                                       **object_states})

        # Transition model
        self.detection_models = init_detection_models(self.agent_config)
        self.no_look = self.agent_config["robot"].get("no_look", True)
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look, pos_precision=0.0001, rot_precision=0.0001)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # Also make an observation model
        robot = self.agent_config["robot"]
        self.localization_model = interpret_localization_model(robot)
        self.robot_observation_model = RobotObservationModel(
            robot['id'], localization_model=self.localization_model)
        self.observation_model = GMOSObservationModel(
            robot["id"], self.detection_models,
            robot_observation_model=self.robot_observation_model,
            no_look=self.no_look)

        # Reward model
        target_ids = self.agent_config["targets"]
        reward_model = GoalBasedRewardModel(target_ids, robot_id=self.robot_id)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    def reachable(self, pos):
        return True  # the real world has no bound

    def provide_observation(self, action=None):
        # will use own observation model.
        observation = self.observation_model.sample(self.state, action)
        # convert voxels into object detections -- what a robot would receive
        real_zobjs = {self.robot_id: observation.z(self.robot_id)}
        for objid in observation:
            if objid == self.robot_id:
                continue
            if isinstance(observation.z(objid), ObjectVoxel):
                voxel = observation.z(objid)
                if voxel.label == objid:
                    objsizes = self.object_spec(objid).get("sizes", [0.12, 0.12, 0.12])
                    zobj = ObjectDetection(objid, voxel.loc, sizes=objsizes)
                else:
                    zobj = ObjectDetection(objid, ObjectDetection.NULL)
                real_zobjs[objid] = zobj

        return GMOSObservation(real_zobjs)

    def object_spec(self, objid):
        return self.agent_config["objects"][objid]


class SimpleSimEnvROSNode:
    """
    note that all messages that this node receives
    should be in the world frame.

    Subscribes:
      ~pomdp_action (KeyValAction)
    Publishes:
      ~state_markers (MarkerArray)
      ~pomdp_observation (KeyValObservation)
    """
    def __init__(self, node_name="simple_sim_env"):
        self.name = "simple_sim_env"
        rospy.init_node(self.name)
        config = rospy.get_param("~config")  # access parameters together as a dictionary
        state_pub_rate = rospy.get_param("~state_pub_rate", 10)
        observation_pub_rate = rospy.get_param("~observation_pub_rate", 3)
        self.world_frame = rospy.get_param("~world_frame")  # fixed frame of the world
        self.br = TransformBroadcaster()

        action_topic = "~pomdp_action"
        state_markers_topic = "~state_markers"
        observation_topic = "~pomdp_observation"
        self.env = SimpleSimEnv(config)
        self.action_sub = rospy.Subscriber(action_topic, KeyValAction, self._action_cb)

        self.state_markers_pub = rospy.Publisher(
            state_markers_topic, MarkerArray, queue_size=10)
        self.state_pub_rate = state_pub_rate

        # observation
        self.observation_pub = rospy.Publisher(observation_topic, KeyValObservation, queue_size=10)
        self.observation_pub_rate = observation_pub_rate

        # navigation related
        self.nav_step_duration = rospy.get_param("~step_duration", 0.1)  # amount of time to execute one step
        self.translation_step_size = rospy.get_param("~translation_step_size", 0.1)  # in meters
        self.rotation_step_size = rospy.get_param("~rotation_step_size", 5)  # in degreesw
        assert self.translation_step_size > 0, "translation_step_size must be > 0"
        assert self.rotation_step_size > 0, "rotation_step_size must be > 0"
        self._navigating = False
        self._action_done_pub = rospy.Publisher("~action_done", String, queue_size=10, latch=True)  # publishes when action is done

    def run(self):
        rospy.loginfo("publishing observations")
        rospy.Timer(rospy.Duration(1/self.observation_pub_rate),
                    lambda event: self.publish_observation())

        print(self.env.state)
        rospy.loginfo("publishing state markers")
        rate = rospy.Rate(self.state_pub_rate)
        while not rospy.is_shutdown():
            state_markers_msg, tf2_msgs = self._make_state_markers_and_tf2msgs(self.env.state)
            self.state_markers_pub.publish(state_markers_msg)
            for t in tf2_msgs:
                self.br.sendTransform(t)
            rate.sleep()

    def publish_observation(self):
        observation = self.env.provide_observation()
        keys = []
        values = []
        # First, make robot observation
        o_robot = observation.z(self.env.robot_id)
        keys.extend(["robot_id", "robot_pose", "objects_found"])
        values.extend([self.env.robot_id,
                       str(o_robot.pose),
                       str(o_robot.objects_found)])

        for objid in observation:
            if objid == self.env.robot_id:
                continue
            zobj = observation.z(objid)
            keys.extend([f"loc_{objid}", f"sizes_{objid}"])
            values.extend([str(zobj.loc), str(zobj.sizes)])
        obs_msg = KeyValObservation(stamp=rospy.Time.now(),
                                    type="joint",
                                    keys=keys,
                                    values=values)
        self.observation_pub.publish(obs_msg)

    @property
    def navigating(self):
        return self._navigating

    def _action_cb(self, action_msg):
        rospy.loginfo("received action to execute")
        if action_msg.type == "nav":
            if not self._navigating:
                kv = {action_msg.keys[i]: action_msg.values[i] for i in range(len(action_msg.keys))}
                goal_id = kv["goal_id"]
                goal_x = float(kv["goal_x"])
                goal_y = float(kv["goal_y"])
                goal_z = float(kv["goal_z"])
                goal_qx = float(kv["goal_qx"])
                goal_qy = float(kv["goal_qy"])
                goal_qz = float(kv["goal_qz"])
                goal_qw = float(kv["goal_qw"])
                goal = (goal_x, goal_y, goal_z, goal_qx, goal_qy, goal_qz, goal_qw)
                rospy.loginfo(f"navigation to {goal}")
                self._navigating = True
                self.navigate_to(goal)
                self._navigating = False
                self._action_done_pub.publish(String(data=f"nav to {goal_id} done."))

            else:
                rospy.loginfo(f"navigation is in progress. Goal ignored.")

        elif action_msg.type == "find":
            self.find()
            time.sleep(0.1)
            self._action_done_pub.publish(String(data=f"find action is done."))

    def _make_state_markers_and_tf2msgs(self, state):
        markers = []
        tf2msgs = []
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        for objid in state.object_states:
            if objid == self.env.robot_id:
                continue  # we draw the robot separately
            sobj = state.s(objid)
            viz_type_name = self.env.object_spec(objid).get("viz_type", "cube")
            viz_type = eval(f"Marker.{viz_type_name.upper()}")
            color = self.env.object_spec(objid).get("color", [0.0, 0.8, 0.0, 0.8])
            objsizes = self.env.object_spec(objid).get("sizes", [0.12, 0.12, 0.12])
            obj_marker = ros_utils.make_viz_marker_for_object(
                sobj.id, sobj.loc, header, viz_type=viz_type,
                color=color, lifetime=1.0, scale=Vector3(x=objsizes[0],
                                                         y=objsizes[1],
                                                         z=objsizes[2]))
            markers.append(obj_marker)
            # get a tf transform from world to object
            tobj = ros_utils.tf2msg_from_object_loc(
                sobj.loc, self.world_frame, objid)
            tf2msgs.append(tobj)

        srobot = state.s(self.env.robot_id)
        robot_marker, trobot = ros_utils.viz_msgs_for_robot_pose(
            srobot.pose, self.world_frame, self.env.robot_id,
            color=[0.9, 0.1, 0.1, 0.9], lifetime=1.0,
            scale=Vector3(x=0.6, y=0.08, z=0.08))
        markers.append(robot_marker)
        # get a tf transform from world to robot
        tf2msgs.append(trobot)
        return MarkerArray(markers), tf2msgs

    def find(self):
        """calls the find action"""
        action = FindAction()
        self.env.state_transition(action, execute=True)


    def navigate_to(self, goal_pose):
        """Super simple navigation. Will first level the camera,
        then rotate the camera in yaw towards the goal, and then
        go in a straightline, and finally rotate it to the goal rotation."""
        current_pose = self.env.state.s(self.env.robot_id).pose
        current_orientation = quat_to_euler(*current_pose[3:])
        goal_orientation = quat_to_euler(*goal_pose[3:])

        # rotate thx
        dthx_actions = self._axis_actions_towards(current_orientation[0], goal_orientation[0], "dthx", 0)
        # rotate thy
        dthy_actions = self._axis_actions_towards(current_orientation[1], goal_orientation[1], "dthy", 1)
        # rotate thz
        dthz_actions = self._axis_actions_towards(current_orientation[2], goal_orientation[2], "dthz", 2)

        # move forward in a straight line over x-y plane.
        dxy_actions = self._xyplane_forward_actions_towards(current_pose[:2], goal_pose[:2], goal_orientation[2])

        # adjust height
        dz_actions = self._axis_actions_towards(current_pose[2], goal_pose[2], "dz", 2)
        all_actions = dthx_actions + dthy_actions + dthz_actions + dxy_actions + dz_actions

        rate = rospy.Rate(1./self.nav_step_duration)
        for action in all_actions:
            self.env.state_transition(action, execute=True)
            rospy.loginfo(f"navigating ({action.name}) ... current pose: {self.env.state.s(self.env.robot_id).pose}")
            rate.sleep()

    def _axis_actions_towards(self, curval, desval, dtype, coord_index):
        """Generates a list of MotionAction3D objects that moves one
        coordinate from current value to desired value, that increments
        by certain step size"""
        actions = []
        diffval = desval - curval
        if dtype in {"dx", "dy", "dz"}:
            step_size = self.translation_step_size
            dapply_index = 0
        else:
            step_size = self.rotation_step_size
            dapply_index = 1
        num_dsteps = int(abs(diffval // step_size))
        dstep = [0, 0, 0]
        if diffval > 0:
            dstep[coord_index] = step_size
        else:
            dstep[coord_index] = -step_size

        dmotion = [(0, 0, 0), (0, 0, 0)]
        dmotion[dapply_index] = tuple(dstep)
        dstep_action = MotionAction3D(tuple(dmotion), motion_name=f"move-{dtype}")
        actions.extend([dstep_action]*num_dsteps)

        # there may be remaining bit, will add one more if needed
        remain_diff = abs(diffval) - num_dsteps*step_size
        if remain_diff > 0:
            dstep_last = [0,0,0]
            if diffval > 0:
                dstep_last[coord_index] = remain_diff
            else:
                dstep_last[coord_index] = -remain_diff
            dmotion_last = [(0, 0, 0), (0, 0, 0)]
            dmotion_last[dapply_index] = tuple(dstep_last)
            dstep_action_last = MotionAction3D(tuple(dmotion_last), motion_name=f"move-{dtype}")
            actions.append(dstep_action_last)
        return actions

    def _xyplane_forward_actions_towards(self, current_xy_pos, goal_xy_pos, goal_yaw):
        """Assume that the agent has finished rotating; Just need to move in the
        goal_yaw direction. 'goal_yaw' should be in degrees"""
        actions = []

        dx = self.translation_step_size*math.cos(to_rad(goal_yaw))
        dy = self.translation_step_size*math.sin(to_rad(goal_yaw))
        distance = euclidean_dist(current_xy_pos, goal_xy_pos)
        num_dsteps = int(distance // self.translation_step_size)
        dmotion = ((dx, dy, 0), (0, 0, 0))
        dstep_action = MotionAction3D(dmotion, motion_name=f"move-forward")
        actions.extend([dstep_action]*num_dsteps)

        remaining = distance - self.translation_step_size*num_dsteps
        if remaining > 0:
            dx_last = remaining*math.cos(to_rad(goal_yaw))
            dy_last = remaining*math.sin(to_rad(goal_yaw))
            dmotion_last = ((dx_last, dy_last, 0), (0, 0, 0))
            dstep_action_last = MotionAction3D(dmotion_last, motion_name=f"move-forward")
            actions.append(dstep_action_last)
        return actions


def main():
    n = SimpleSimEnvROSNode()
    n.run()

if __name__ == "__main__":
    main()
