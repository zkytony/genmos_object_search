#!/usr/bin/env python

import rospy
import diagnostic_msgs
from pomdp_py.utils import typ
from sloop_object_search.msg import KeyValAction
from sloop_object_search.ros.framework import ActionExecutor
from sloop_object_search.oopomdp.domain.action import (MotionActionTopo,
                                                       StayAction,
                                                       FindAction,
                                                       MotionAction2D)
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic2D
from sloop_object_search.utils.math import to_rad
from actionlib_msgs.msg import GoalStatus

from bosdyn.client.math_helpers import Quat
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api import robot_command_pb2
from bosdyn.api.geometry_pb2 import Vec2, Vec3, SE2VelocityLimit, SE2Velocity
import rbd_spot

# distance between hand and body frame origin
SPOT_HAND_TO_BODY_DISTANCE = 0.65


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


    @classmethod
    def action_to_ros_msg(self, agent, action, goal_id):
        print(action)

        if isinstance(action, MotionActionTopo):
            # The navigation goal is specified with respect to the global map frame.
            goal_pos = action.dst_pose[:2]
            goal_yaw = to_rad(action.dst_pose[2])
            metric_pos = agent.grid_map.to_metric_pos(*goal_pos)
            action_msg = KeyValAction(stamp=rospy.Time.now(),
                                      type="move_topo",
                                      keys=["goal_x", "goal_y", "goal_yaw", "name"],
                                      values=[str(metric_pos[0]), str(metric_pos[1]), str(goal_yaw), action.name])
            return action_msg

        elif isinstance(action, MotionAction2D):
            # We will make the robot move its arm with body follow; The robot arm
            # movement is specified with respect to the robot frame.
            robot_pose_after_action = RobotTransBasic2D.transform_pose((0, 0, 0.0), action)

            # note that by default the gripper is a bit forward with respect to the body origin
            metric_pos_x = 0.65 + robot_pose_after_action[0] * agent.grid_map.grid_size
            metric_pos_y = robot_pose_after_action[1] * agent.grid_map.grid_size
            metric_yaw = to_rad(-robot_pose_after_action[2])  # For spot's frame, we needed to reverse the angle
            action_msg = KeyValAction(stamp=rospy.Time.now(),
                                      type="move_2d",
                                      keys=["goal_x", "goal_y", "goal_yaw", "name"],
                                      values=[str(metric_pos_x), str(metric_pos_y), str(metric_yaw), action.name])
            return action_msg

        elif isinstance(action, FindAction):
            return KeyValAction(stamp=rospy.Time.now(),
                                type="find",
                                keys=[],
                                values=[])


    def _execute_action_cb(self, msg):
        if msg.type == "nothing":
            return

        kv = {msg.keys[i]: msg.values[i] for i in range(len(msg.keys))}
        # used to identify this action as a goal for execution
        action_id = ActionExecutor.action_id(msg)
        rospy.loginfo("received action to execute")
        if msg.type == "move_topo":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_yaw = float(kv["goal_yaw"])
            goal = (goal_x, goal_y, goal_yaw)
            self.publish_status(GoalStatus.ACTIVE,
                                typ.info(f"executing navigation goal {kv['name']}"),
                                action_id, msg.stamp)
            rbd_spot.arm.close_gripper(self.conn, self.command_client)
            rbd_spot.arm.stow(self.conn, self.command_client)
            nav_feedback_code = rbd_spot.graphnav.navigateTo(
                self.conn, self.graphnav_client, goal,
                tolerance=(0.25, 0.25, 0.15),
                speed=None,
                travel_params=graph_nav_pb2.TravelParams(max_distance=0.15,   # more lenient
                                                         disable_alternate_route_finding=True))
            self.publish_nav_status(nav_feedback_code, action_id, msg.stamp)

        elif msg.type == "move_2d":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_z = 0.25  # fixed height (2d)
            goal_yaw = float(kv["goal_yaw"])
            goal_quat = Quat.from_yaw(goal_yaw)
            goal = (goal_x, goal_y, goal_z, goal_quat.x,
                    goal_quat.y, goal_quat.z, goal_quat.w)
            self.publish_status(GoalStatus.ACTIVE,
                                typ.info(f"executing {kv['name']} with moveEE with body follow"),
                                action_id, msg.stamp)
            rbd_spot.arm.open_gripper(self.conn, self.command_client)
            cmd_success = rbd_spot.arm.moveEEToWithBodyFollow(
                self.conn, self.command_client, self.robot_state_client, goal)
            # also, rotate the body a little bit; TODO: ad-hoc
            if "TurnLeft" in kv['name']:
                rbd_spot.body.velocityCommand(
                    self.conn, self.command_client, 0.0, 0.0, 0.5, duration=1.0)  # 1s is roughtly ~<45deg
            elif "TurnRight" in kv['name']:
                rbd_spot.body.velocityCommand(
                    self.conn, self.command_client, 0.0, 0.0, -0.5, duration=1.0)
            if cmd_success:
                self.publish_status(GoalStatus.SUCCEEDED,
                                    typ.success("arm movement succeeded"),
                                    action_id, msg.stamp)
            else:
                self.publish_status(GoalStatus.ABORTED,
                                    typ.error("arm movement failed"),
                                    action_id, msg.stamp)

        elif msg.type == "find":
            # signal find action with a bit of arm motion and then stow.
            rbd_spot.arm.moveEETo(
                self.conn, self.command_client, self.robot_state_client, (0.65, 0.0, 0.35))
            rbd_spot.arm.stow(self.conn, self.command_client)
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("find action succeeded"),
                                action_id, msg.stamp)

        elif msg.type == "stow_arm":
            rbd_spot.arm.stow(self.conn, self.command_client)
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("arm stowed"),
                                action_id, msg.stamp)


    def publish_nav_status(self, nav_feedback_code, action_id, stamp):
        nav_status = self.graphnav_client.navigation_feedback(nav_feedback_code)
        if nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            self.publish_status(GoalStatus.SUCCEEDED,
                                typ.success("navigation succeeded"),
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot got lost when navigating the route"),
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot got stuck when navigating the route"),
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self.publish_status(GoalStatus.ABORTED,
                                typ.error("Robot is impaired."),
                                action_id, stamp)
        else:
            self.publish_status(GoalStatus.PENDING,
                                "navigation command is not complete yet",
                                action_id, stamp)
