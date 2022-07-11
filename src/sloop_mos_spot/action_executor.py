#!/usr/bin/env python

import rospy
import diagnostic_msgs
from sloop_ros.msg import KeyValAction
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
import rbd_spot


class SpotSloopActionExecutor(ActionExecutor):

    def __init__(self):
        super().__init__()
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
            goal_pos = action.dst_pose[:2]
            goal_yaw = to_rad(action.dst_pose[2])
            metric_pos = agent.grid_map.to_metric_pos(*goal_pos)
            action_msg = KeyValAction(stamp=rospy.Time.now(),
                                      type="move_topo",
                                      keys=["goal_x", "goal_y", "goal_yaw", "name"],
                                      values=[str(metric_pos[0]), str(metric_pos[1]), str(goal_yaw), action.name])
            return action_msg

        elif isinstance(action, MotionAction2D):
            # the motion is with respect to the robot's current pose in the grid map.
            # we will interpret that as a map frame end effector pose, which will
            # be accomplished by moving the arm with body follow.
            current_robot_pose = agent.belief.mpe().s(agent.robot_id)['pose']
            robot_pose_after_action = RobotTransBasic2D.transform_pose(current_robot_pose, action)
            goal_pos = robot_pose_after_action[:2]
            metric_pos = agent.grid_map.to_metric_pos(*goal_pos)
            goal_yaw = to_rad(robot_pose_after_action[2])
            action_msg = KeyValAction(stamp=rospy.Time.now(),
                                      type="move_2d",
                                      keys=["goal_x", "goal_y", "goal_yaw", "name"],
                                      values=[str(metric_pos[0]), str(metric_pos[1]), str(goal_yaw), action.name])
            return action_msg


    def _execute_action_cb(self, msg):
        kv = {msg.keys[i]: msg.values[i] for i in range(len(msg.keys))}
        # used to identify this action as a goal for execution
        action_id = "{}-{}".format(msg.type, str(msg.stamp))
        if msg.type == "move_topo":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_yaw = float(kv["goal_yaw"])
            goal = (goal_x, goal_y, goal_yaw)
            self.publish_status(GoalStatus.ACTIVE,
                                f"executing navigation goal {kv['name']}",
                                action_id, msg.stamp)
            nav_feedback_code = rbd_spot.graphnav.navigateTo(
                self.conn, self.graphnav_client, goal,
                tolerance=(0.25, 0.25, 0.15), speed="slow")

            self.publish_nav_status(nav_feedback_code, action_id, msg.stamp)

        elif msg.type == "move_2d":
            goal_x = float(kv["goal_x"])
            goal_y = float(kv["goal_y"])
            goal_z = 0.25  # fixed height (2d)
            goal_yaw = float(kv["goal_yaw"])
            goal_quat = Quat.from_yaw(goal_yaw)
            goal = (goal_x, goal_y, goal_z, *goal_quat)
            self.publish_status(GoalStatus.ACTIVE,
                                f"executing moveEE with body follow action {kv['name']}",
                                action_id, msg.stamp)
            cmd_success = rbd_spot.arm.moveToEEWithBodyFollow(
                self.conn, self.command_client, self.robot_state_client, goal)
            if cmd_success:
                self.publish_status(GoalStatus.SUCCEEDED,
                                    "arm movement succeeded",
                                    action_id, msg.stamp)
            else:
                self.publish_status(GoalStatus.ABORTED,
                                    "arm movement failed",
                                    action_id, msg.stamp)


    def publish_nav_status(self, nav_feedback_code, action_id, stamp):
        nav_status = self.graphnav_client.navigation_feedback(nav_feedback_code)
        if nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            self.publish_status(GoalStatus.SUCCEEDED,
                                "navigation succeeded",
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self.publish_status(GoalStatus.ABORTED,
                                "Robot got lost when navigating the route",
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self.publish_status(GoalStatus.ABORTED,
                                "Robot got stuck when navigating the route",
                                action_id, stamp)
        elif nav_status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self.publish_status(GoalStatus.ABORTED,
                                "Robot is impaired.",
                                action_id, stamp)
        else:
            self.publish_status(GoalStatus.PENDING,
                                "navigation command is not complete yet",
                                action_id, stamp)
