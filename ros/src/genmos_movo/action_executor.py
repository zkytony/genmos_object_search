#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg as geometry_msgs

import sys
import diagnostic_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
from actionlib_msgs.msg import GoalStatus

from pomdp_py.utils import typ

from genmos_object_search_ros.msg import KeyValAction
from genmos_ros.action_executor import ActionExecutor
from visualization_msgs.msg import Marker, MarkerArray
from genmos_ros import ros_utils
from genmos_object_search.utils.misc import confirm_yes
import genmos_object_search.utils.math as math_utils

from .head_jtas import HeadJTAS
from .torso_jtas import TorsoJTAS
from .waypoint import WaypointApply

TORSO_HEIGHT_MAX = 0.5
TORSO_HEIGHT_MIN = 0.0
CAMERA_MIN_HEIGHT = 1.04  # when torso is 0.0

MAX_PAN = 45
MIN_PAN = -45
MAX_TILT = 45
MIN_TILT = -45


class MovoGenMOSActionExecutor(ActionExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._robot_pose_topic = "~robot_pose"
        self.robot_id = rospy.get_param("~robot_id")
        self.world_frame = rospy.get_param("~world_frame")
        self.camera_frame = "kinect2_color_frame"
        self.body_frame = "base_link"
        self.check_before_execute = rospy.get_param("~check_before_execute", True)
        self._goal_viz_pub = rospy.Publisher(
            "~goal_markers", MarkerArray, queue_size=10, latch=True)
        self.torso_jtas = TorsoJTAS()
        self.head_jtas = HeadJTAS()

    def _execute_action_cb(self, msg):
        if msg.type == "nothing":
            return

        kv = {msg.keys[i]: msg.values[i] for i in range(len(msg.keys))}
        # used to identify this action as a goal for execution
        action_id = kv["action_id"]
        rospy.loginfo("received action to execute")

        if msg.type == "nav":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_z = float(kv["goal_z"])
            goal_qx = float(kv["goal_qx"])
            goal_qy = float(kv["goal_qy"])
            goal_qz = float(kv["goal_qz"])
            goal_qw = float(kv["goal_qw"])
            nav_type = kv["nav_type"]
            self.publish_status(GoalStatus.ACTIVE,
                                typ.info(f"executing {kv['action_id']}..."),
                                action_id, msg.stamp)
            success = self.move_viewpoint(action_id, nav_type,
                                          (goal_x, goal_y, goal_z,
                                           goal_qx, goal_qy, goal_qz, goal_qw))
            if success:
                self.publish_status(GoalStatus.SUCCEEDED,
                                    typ.success("move viewpoint succeeded"),
                                    action_id, msg.stamp, pub_done=True)
            else:
                self.publish_status(GoalStatus.ABORTED,
                                    typ.error("move viewpoint failed"),
                                    action_id, msg.stamp, pub_done=True)

        elif msg.type == "find":
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("find action succeeded"),
                                action_id, msg.stamp, pub_done=True)


    def move_viewpoint(self, action_id, nav_type, goal_pose):
        if nav_type not in {"2d", "3d"}:
            raise ValueError("nav_type should be '2d' or '3d'")

        # Visualize the goal.
        # clear markers first
        clear_msg = ros_utils.clear_markers(std_msgs.Header(stamp=rospy.Time.now(),
                                                            frame_id=self.world_frame), ns="")
        self._goal_viz_pub.publish(clear_msg)

        # then make the markers for goals; We will still visualize hand messages (that's what we care about)
        _marker_scale = geometry_msgs.Vector3(x=0.4, y=0.05, z=0.05)
        _goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id + "_goal", goal_pose, std_msgs.Header(frame_id=self.world_frame, stamp=rospy.Time.now()),
            scale=_marker_scale, color=[0.53, 0.95, 0.99, 0.9], lifetime=0)
        self._goal_viz_pub.publish(MarkerArray([_goal_marker_msg]))

        # Check if the user wants to execute
        if self.check_before_execute:
            if not confirm_yes("Execute change viewpoint action?"):
                rospy.loginfo(typ.warning("User terminates action execution"))
                return False

        # reset head and torso
        self.reset_torso()
        self.reset_head()

        # Note that the goal pose is the camera pose in the world frame.
        # But when we do navigation, we control the robot's body. So, similar
        # to Spot's execution, we first move the body to the goal 2D location
        # with desired yaw, then adjust torso and head to look in the right
        # direction.
        _, _, yaw = math_utils.quat_to_euler(*goal_pose[3:])
        x, y = goal_pose[:2]
        _waypoint_nav = WaypointApply((x, y, 0.5),
                                      math_utils.euler_to_quat(0, 0, yaw),
                                      xy_tolerance=0.2,
                                      rot_tolerance=math_utils.to_rad(30))
        # navigation sometimes gets close but not reaching the goal;
        # but it doesn't hurt moving torso and turning the head
        if _waypoint_nav.status != WaypointApply.Status.SUCCESS:
            return False

        # We don't have a simple way to control the robot to move back without
        # guaranteeing that it doens't hit something (unlike spot). So we will
        # just not do that, and move the torso and tilt the camera
        height = (goal_pose[2] - CAMERA_MIN_HEIGHT)
        torso_goal = self.make_torso_goal(height=height)
        self.torso_jtas.client.send_goal(torso_goal)
        rospy.loginfo("Waiting for torso action to finish")
        self.torso_jtas.client.wait_for_result(timeout=rospy.Duration(20))

        # We then pan/tilt the camera as needed
        robot_pose_msg = ros_utils.WaitForMessages(
            [self._robot_pose_topic], [geometry_msgs.PoseStamped],
            verbose=True).messages[0]
        robot_pose_tuple = ros_utils.pose_to_tuple(robot_pose_msg.pose)

        _, goal_pitch, goal_yaw = math_utils.quat_to_euler(*goal_pose[3:])
        _, _, cur_yaw = math_utils.quat_to_euler(*robot_pose_tuple[3:])
        head_goal = self.make_head_goal(pan=goal_yaw - cur_yaw,
                                        tilt=-goal_pitch)
        self.head_jtas.client.send_goal(head_goal)
        self.head_jtas.client.wait_for_result(timeout=rospy.Duration(20))
        return True

    def make_torso_goal(self, **kwargs):
        vel = kwargs.get("vel", 0.05)
        desired_height = kwargs["height"]
        if desired_height < TORSO_HEIGHT_MIN or desired_height > TORSO_HEIGHT_MAX:
            rospy.logwarn("Specified torso goal height {} is out of range ({} ~ {}). Will clamp."\
                          .format(desired_height, TORSO_HEIGHT_MIN, TORSO_HEIGHT_MAX))
            desired_height = max(TORSO_HEIGHT_MIN, min(TORSO_HEIGHT_MAX, desired_height))
        return TorsoJTAS.make_goal(desired_height,
                                   v=vel)

    def make_head_goal(self, **kwargs):
        vel = kwargs.get("vel", 0.3)
        # limit pan and tilt ranges
        pan = max(MIN_PAN, min(MAX_PAN, kwargs["pan"]))
        tilt = max(MIN_TILT, min(MAX_TILT, kwargs["tilt"]))
        desired_pan = math_utils.to_radians(pan)
        desired_tilt = math_utils.to_radians(tilt)
        return HeadJTAS.make_goal(desired_pan, desired_tilt, v=vel)

    def reset_head(self):
        head_goal = self.make_head_goal(pan=0, tilt=-30)
        self.head_jtas.client.send_goal(head_goal)
        self.head_jtas.client.wait_for_result(timeout=rospy.Duration(20))

    def reset_torso(self):
        torso_goal = self.make_torso_goal(height=0.2)
        self.torso_jtas.client.send_goal(torso_goal)
        self.torso_jtas.client.wait_for_result(timeout=rospy.Duration(20))
