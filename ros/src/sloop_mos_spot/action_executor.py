#!/usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg as geometry_msgs

import sys
import diagnostic_msgs
from pomdp_py.utils import typ

from sloop_object_search_ros.msg import KeyValAction
from sloop_object_search.oopomdp.domain.action import (MotionActionTopo,
                                                       StayAction,
                                                       FindAction,
                                                       MotionAction2D)
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic2D
from sloop_mos_ros.action_executor import ActionExecutor
from sloop_mos_ros import ros_utils
from visualization_msgs.msg import Marker, MarkerArray
from actionlib_msgs.msg import GoalStatus
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import sloop_object_search.utils.math as math_utils
from sloop_object_search.utils.misc import confirm_yes

from bosdyn.client.math_helpers import Quat
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api import robot_command_pb2
from bosdyn.api.geometry_pb2 import Vec2, Vec3, SE2VelocityLimit, SE2Velocity, Quaternion
from bosdyn.client.exceptions import InvalidRequestError
import rbd_spot

# distance between hand and body frame origin
SPOT_HAND_TO_BODY_DISTANCE = 0.65
LONG_NAV_DISTANCE = 2.0
NAV_HEIGHT = 0.25

class SpotSloopActionExecutor(ActionExecutor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We do want lease access
        self.conn = rbd_spot.SpotSDKConn(sdk_name="SpotSloopActionExecutorClient",
                                         acquire_lease=True,
                                         take_lease=True)
        self.graphnav_client = rbd_spot.graphnav.create_client(self.conn)
        self.robot_state_client = rbd_spot.state.create_client(self.conn)
        self.command_client = rbd_spot.arm.create_client(self.conn)

        self._robot_pose_topic = "~robot_pose"
        self.robot_id = rospy.get_param("~robot_id")
        self.world_frame = rospy.get_param("~world_frame")
        self.hand_frame = "hand"
        self.body_frame = "body"
        self.check_before_execute = rospy.get_param("~check_before_execute", True)
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)

        self._goal_viz_pub = rospy.Publisher(
            "~goal_markers", MarkerArray, queue_size=10, latch=True)


    def move_viewpoint(self, action_id, goal_pose):
        # To navigate to this pose, we first navigate to a pose
        # at the same position, with the same yaw, but a set height,
        # and zero in other rotations.
        rbd_spot.arm.close_gripper(self.conn, self.command_client)
        rbd_spot.arm.stow(self.conn, self.command_client)

        # Compute navigation goal and the final goal. Note that the goal is
        # for the hand (though in world frame), but navigateTo controls the body. We will account for this.
        thx, thy, thz = math_utils.quat_to_euler(*goal_pose[3:])
        nav_quat = math_utils.euler_to_quat(0, 0, thz)
        nav_goal = (goal_pose[0], goal_pose[1], NAV_HEIGHT, *nav_quat)
        nav_goal_stamped = ros_utils.pose_tuple_to_pose_stamped(goal_pose, self.world_frame)
        _trans = ros_utils.tf2_lookup_transform(self.tfbuffer, self.body_frame, self.hand_frame, rospy.Time(0))
        nav_goal_body_stamped = ros_utils.tf2_do_transform(nav_goal_stamped, _trans)
        nav_goal_body = ros_utils.pose_tuple_from_pose_stamped(nav_goal_body_stamped)

        # will always set the goal pose's roll to 0. Just pitch and yaw is enough!
        goal_quat = math_utils.euler_to_quat(0, thy, thz)
        goal_pose = (*goal_pose[:3], *goal_quat)
        goal_pose_msg = ros_utils.pose_tuple_to_pose_stamped(goal_pose, self.world_frame)
        _trans = ros_utils.tf2_lookup_transform(self.tfbuffer, self.hand_frame, self.world_frame, rospy.Time(0))
        goal_pose_hand_msg = ros_utils.tf2_do_transform(goal_pose_msg, _trans)
        _trans = ros_utils.tf2_lookup_transform(self.tfbuffer, self.body_frame, self.hand_frame, rospy.Time(0))
        goal_pose_body_msg = ros_utils.tf2_do_transform(goal_pose_hand_msg, _trans)

        goal_pose_hand = ros_utils.pose_tuple_from_pose_stamped(goal_pose_hand_msg)
        goal_pose_body = ros_utils.pose_tuple_from_pose_stamped(goal_pose_body_msg)

        # Publish visualization markers
        # clear markers first
        clear_msg = ros_utils.clear_markers(std_msgs.Header(stamp=rospy.Time.now(),
                                                            frame_id=self.world_frame), ns="")
        self._goal_viz_pub.publish(clear_msg)
        # then make the markers for goals; We will still visualize hand messages (that's what we care about)
        _marker_scale = geometry_msgs.Vector3(x=0.4, y=0.05, z=0.05)
        _nav_goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id + "_nav", nav_goal, std_msgs.Header(frame_id=self.world_frame, stamp=rospy.Time.now()),
            scale=_marker_scale, color=[0.53, 0.95, 0.99, 0.9], lifetime=0)
        _goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id, goal_pose_hand, std_msgs.Header(frame_id=self.hand_frame, stamp=rospy.Time.now()),
            scale=_marker_scale, color=[0.12, 0.89, 0.95, 0.9], lifetime=0)
        self._goal_viz_pub.publish(MarkerArray([_nav_goal_marker_msg,
                                                _goal_marker_msg]))

        if self.check_before_execute:
            if not confirm_yes("Execute change viewpoint action?"):
                rospy.loginfo(typ.warning("User terminates action execution"))
                return False

        # Navigate to navigation pose
        nav_feedback_code = rbd_spot.graphnav.navigateTo(
            self.conn, self.graphnav_client, nav_goal_body,
            tolerance=(0.25, 0.25, 0.15),
            speed="medium",
            travel_params=graph_nav_pb2.TravelParams(max_distance=0.15,   # more lenient
                                                     disable_alternate_route_finding=True))
        nav_success = self.publish_nav_status(nav_feedback_code, action_id, rospy.Time.now())
        rate = rospy.Rate(2.0)
        while nav_success is None:
            nav_success = self.publish_nav_status(nav_feedback_code, action_id, rospy.Time.now())
            rate.sleep()
        if not nav_success:
            return False

        # The robot sometimes doesn't rotate, because it starts out too close
        # to the navigation goal position and already satisfies it . So we will
        # fulfill the yaw in goal by rotating manually
        current_pose_msg = ros_utils.WaitForMessages(
            [self._robot_pose_topic], [geometry_msgs.PoseStamped],
            verbose=True).messages[0]
        current_pose = ros_utils.pose_to_tuple(current_pose_msg.pose)
        cur_thz = math_utils.quat_to_euler(*current_pose[3:])[2]
        duration = 0.75
        v_rot = math_utils.to_rad((thz - cur_thz) / duration)
        try:
            rbd_spot.body.velocityCommand(
                self.conn, self.command_client, 0.0, 0.0, v_rot, duration=duration)  # 1s is roughtly ~<45deg
        except InvalidRequestError as ex:
            # this is ok. we don't have to rotate
            rospy.logwarn(str(ex))
            pass

        # Then, we use moveEE with body follow to move the camera to the goal pose.
        # Note that moveEEToWithBodyFollow takes in a pose relative to the
        # current body frame, while our goal_pose is in the world frame.
        # open gripper to better see
        rbd_spot.arm.open_gripper(self.conn, self.command_client)
        rbd_spot.arm.unstow(self.conn, self.command_client)

        cmd_success = rbd_spot.arm.moveEETo(
            self.conn, self.command_client, self.robot_state_client, goal_pose_body)
        return cmd_success


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
            self.publish_status(GoalStatus.ACTIVE,
                                typ.info(f"executing {kv['action_id']}..."),
                                action_id, msg.stamp)
            success = self.move_viewpoint(action_id,
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
            # signal find action by moving forward a little bit
            cmd_success = rbd_spot.arm.moveEEToWithBodyFollow(
                self.conn, self.command_client, self.robot_state_client, (0.1, 0.0, 0.0))
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("find action succeeded"),
                                action_id, msg.stamp, pub_done=True)


    def publish_nav_status(self, nav_feedback_code, action_id, stamp):
        nav_status = self.graphnav_client.navigation_feedback(nav_feedback_code)
        if nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("navigation succeeded"),
                                action_id, stamp)
            return True
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot got lost when navigating the route"),
                                action_id, stamp)
            return False
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot got stuck when navigating the route"),
                                action_id, stamp)
            return False
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot is impaired."),
                                action_id, stamp)
            return False
        else:
            self.publish_status(GoalStatus.PENDING,
                                "navigation command is not complete yet",
                                action_id, stamp)
            return None
