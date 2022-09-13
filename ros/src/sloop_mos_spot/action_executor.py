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

from bosdyn.client.math_helpers import Quat
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api import robot_command_pb2
from bosdyn.api.geometry_pb2 import Vec2, Vec3, SE2VelocityLimit, SE2Velocity
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

        # also, subscribes to current robot pose
        self._robot_pose_topic = "~robot_pose"
        self.robot_id = rospy.get_param("~robot_id")
        self.world_frame = rospy.get_param("~world_frame")
        self.body_frame = rospy.get_param("~body_frame")
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)

        self._goal_viz_pub = rospy.Publisher(
            "~goal_markers", MarkerArray, queue_size=10, latch=True)


    def move_viewpoint(self, action_id, goal_pose):
        # To navigate to this pose, we first navigate to a pose
        # at the same position, with the same yaw, but a set height,
        # and zero in other rotations.
        robot_pose_msg = ros_utils.WaitForMessagaes(
            [self._robot_pose_topic], [geometry_msgs.PoseStamped],
            verbose=True).messages[0]
        current_pose = ros_utils.pose_to_tuple(robot_pose_msg.pose)

        # If the distance between the two positions is long,
        # the robot should stow its arm
        if math_utils.euclidean_dist(current_pose[:3], goal_pose[:3]) >= LONG_NAV_DISTANCE:
            rbd_spot.arm.close_gripper(self.conn, self.command_client)
            rbd_spot.arm.stow(self.conn, self.command_client)

        # Compute navigation goal and the final goal
        nav_pos = ()
        thx, thy, thz = math_utils.quat_to_euler(goal_pose[3:])
        nav_quat = Quat.from_yaw(thz)
        nav_goal = (goal_pose[0], goal_pose[1], NAV_HEIGHT,
                    nav_quat.x, nav_quat.y, nav_quat.z, nav_quat.w)
        nav_goal_msg = ros_utils.pose_tuple_to_pose_stamped(nav_goal, self.world_frame)
        goal_pose_msg = ros_utils.pose_tuple_to_pose_stamped(goal_pose, self.world_frame)
        goal_pose_body_msg = ros_utils.tf2_transform(self.tfbuffer, goal_pose_msg, self.body_frame)
        goal_pose_body = ros_utils.pose_tuple_from_pose_stamped(goal_pose_body_msg)

        # Publish visualization markers
        # clear markers first
        clear_msg = ros_utils.clear_markers(std_msgs.Header(stamp=rospy.Time.now(),
                                                            frame_id=self.world_frame), ns="")
        self._goal_viz_pub.publish(clear_msg)
        # then make the markers for goals
        _marker_scale = geometry_msgs.msg.Vector3(x=0.4, y=0.05, z=0.05)
        _nav_goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id, nav_goal, std_msgs.Header(frame_id=self.world_frame, stamp=rospy.Time.now()),
            scale=_marker_scale, color=[0.53, 0.95, 0.99, 0.9], lifetime=0)
        _goal_marker_msg = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id, goal_pose_body, std_msgs.Header(frame_id=self.body_frame, stamp=rospy.Time.now()),
            scale=_marker_scale, color=[0.12, 0.89, 0.95, 0.9], lifetime=0)
        self._goal_viz_pub.publish(MarkerArray([_nav_goal_marker_msg,
                                                _goal_marker_msg]))
        import pdb; pdb.set_trace()

        # Navigate to navigation pose
        nav_feedback_code = rbd_spot.graphnav.navigateTo(
            self.conn, self.graphnav_client, nav_goal,
            tolerance=(0.25, 0.25, 0.15),
            speed="medium",
            travel_params=graph_nav_pb2.TravelParams(max_distance=0.15,   # more lenient
                                                     disable_alternate_route_finding=True))
        nav_success = self.publish_nav_status(nav_feedback_code, action_id, msg.stamp)
        rate = rospy.Rate(2.0)
        while nav_success is None:
            nav_success = self.publish_nav_status(nav_feedback_code, action_id, msg.stamp)
            rate.sleep()
        if not nav_success:
            return False

        # Then, we use moveEE with body follow to move the camera to the goal pose.
        # Note that moveEEToWithBodyFollow takes in a pose relative to the
        # current body frame, while our goal_pose is in the world frame.
        # open gripper to better see
        rbd_spot.arm.open_gripper(self.conn, self.command_client)

        # get goal pose in body frame
        cmd_success = rbd_spot.arm.moveEEToWithBodyFollow(
            self.conn, self.command_client, self.robot_state_client, goal_pose_body)
        return cmd_success


    def _execute_action_cb(self, msg):
        if msg.type == "nothing":
            return

        kv = {msg.keys[i]: msg.values[i] for i in range(len(msg.keys))}
        # used to identify this action as a goal for execution
        action_id = ActionExecutor.action_id(msg)
        rospy.loginfo("received action to execute")

        if msg.type == "nav":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_z = float(kv["goal_z"])
            goal_qx = float(kv["goal_qx"])
            goal_qy = float(kv["goal_qy"])
            goal_qz = float(kv["goal_qz"])
            goal_qw = float(kv["goal_qw"])
            action_id = kv["action_id"]
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
        # if msg.type == "move_topo":
        #     goal_x = float(kv["goal_x"])
        #     goal_y = float(kv["goal_y"])
        #     goal_yaw = float(kv["goal_yaw"])
        #     goal = (goal_x, goal_y, goal_yaw)
        #     self.publish_status(GoalStatus.ACTIVE,
        #                         typ.info(f"executing navigation goal {kv['name']}"),
        #                         action_id, msg.stamp)
        #     rbd_spot.arm.close_gripper(self.conn, self.command_client)
        #     rbd_spot.arm.stow(self.conn, self.command_client)
        #     nav_feedback_code = rbd_spot.graphnav.navigateTo(
        #         self.conn, self.graphnav_client, goal,
        #         tolerance=(0.25, 0.25, 0.15),
        #         speed=None,
        #         travel_params=graph_nav_pb2.TravelParams(max_distance=0.15,   # more lenient
        #                                                  disable_alternate_route_finding=True))
        #     self.publish_nav_status(nav_feedback_code, action_id, msg.stamp)

        # elif msg.type == "move_2d":
        #     goal_x = float(kv["goal_x"])
        #     goal_y = float(kv["goal_y"])
        #     goal_z = 0.25  # fixed height (2d)
        #     goal_yaw = float(kv["goal_yaw"])
        #     goal_quat = Quat.from_yaw(goal_yaw)
        #     goal = (goal_x, goal_y, goal_z, goal_quat.x,
        #             goal_quat.y, goal_quat.z, goal_quat.w)
        #     self.publish_status(GoalStatus.ACTIVE,
        #                         typ.info(f"executing {kv['name']} with moveEE with body follow"),
        #                         action_id, msg.stamp)
        #     rbd_spot.arm.open_gripper(self.conn, self.command_client)
        #     cmd_success = rbd_spot.arm.moveEEToWithBodyFollow(
        #         self.conn, self.command_client, self.robot_state_client, goal)
        #     # also, rotate the body a little bit; TODO: ad-hoc
        #     if "TurnLeft" in kv['name']:
        #         rbd_spot.body.velocityCommand(
        #             self.conn, self.command_client, 0.0, 0.0, 0.5, duration=1.0)  # 1s is roughtly ~<45deg
        #     elif "TurnRight" in kv['name']:
        #         rbd_spot.body.velocityCommand(
        #             self.conn, self.command_client, 0.0, 0.0, -0.5, duration=1.0)
        #     if cmd_success:
        #         self.publish_status(GoalStatus.SUCCEEDED,
        #                             typ.success("arm movement succeeded"),
        #                             action_id, msg.stamp)
        #     else:
        #         self.publish_status(GoalStatus.ABORTED,
        #                             typ.error("arm movement failed"),
        #                             action_id, msg.stamp)

        elif msg.type == "find":
            # signal find action by closing and opening gripper

            # # signal find action with a bit of arm motion and then stow.
            # rbd_spot.arm.moveEETo(
            #     self.conn, self.command_client, self.robot_state_client, (0.65, 0.0, 0.35))
            # rbd_spot.arm.stow(self.conn, self.command_client)
            # self.publish_status(GoalStatus.SUCCEEDED,
            #                     typ.success("find action succeeded"),
            #                     action_id, msg.stamp)


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



    # @classmethod
    # def action_to_ros_msg(self, agent, action, goal_id):
    #     print(action)

    #     if isinstance(action, MotionActionTopo):
    #         # The navigation goal is specified with respect to the global map frame.
    #         goal_pos = action.dst_pose[:2]
    #         goal_yaw = to_rad(action.dst_pose[2])
    #         metric_pos = agent.grid_map.to_metric_pos(*goal_pos)
    #         action_msg = KeyValAction(stamp=rospy.Time.now(),
    #                                   type="move_topo",
    #                                   keys=["goal_x", "goal_y", "goal_yaw", "name"],
    #                                   values=[str(metric_pos[0]), str(metric_pos[1]), str(goal_yaw), action.name])
    #         return action_msg

    #     elif isinstance(action, MotionAction2D):
    #         # We will make the robot move its arm with body follow; The robot arm
    #         # movement is specified with respect to the robot frame.
    #         robot_pose_after_action = RobotTransBasic2D.transform_pose((0, 0, 0.0), action)

    #         # note that by default the gripper is a bit forward with respect to the body origin
    #         metric_pos_x = 0.65 + robot_pose_after_action[0] * agent.grid_map.grid_size
    #         metric_pos_y = robot_pose_after_action[1] * agent.grid_map.grid_size
    #         metric_yaw = to_rad(-robot_pose_after_action[2])  # For spot's frame, we needed to reverse the angle
    #         action_msg = KeyValAction(stamp=rospy.Time.now(),
    #                                   type="move_2d",
    #                                   keys=["goal_x", "goal_y", "goal_yaw", "name"],
    #                                   values=[str(metric_pos_x), str(metric_pos_y), str(metric_yaw), action.name])
    #         return action_msg

    #     elif isinstance(action, FindAction):
    #         return KeyValAction(stamp=rospy.Time.now(),
    #                             type="find",
    #                             keys=[],
    #                             values=[])
