# Note: adapted from code by Yoon.
# code from MOS3D
# /author: Kaiyu Zheng
import argparse
import sys
from copy import copy
import rospy
import actionlib
import random

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
from control_msgs.msg import JointTrajectoryControllerState

class HeadJTAS(object):
    def __init__(self):
        self._client = actionlib.SimpleActionClient(
            'movo/head_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)

    @property
    def client(self):
        return self._client

    @staticmethod
    def make_goal(desired_pan, desired_tilt,
                  v=0.3):
        """desired_pan, desired_tilt (radian) are angles of pan and tilt joints of the head"""
        def _add_point(goal, positions, time):
            point = JointTrajectoryPoint()
            point.positions = copy(positions)
            point.velocities = [0.0] * len(goal.trajectory.joint_names)
            point.time_from_start = rospy.Duration(time)
            goal.trajectory.points.append(point)

        cur_pan, cur_tilt = HeadJTAS.wait_for_head()
        goal = FollowJointTrajectoryGoal()

        total_time_head = 0.0
        _add_point(goal, [cur_pan, cur_tilt], 0.0)
        # First pan
        if desired_pan < cur_pan:
            vel = -v
        else:
            vel = v
        dt = abs(abs(desired_pan - cur_pan) / vel)
        total_time_head += dt
        _add_point(goal, [desired_pan, cur_tilt],total_time_head)
        # then tilt
        if desired_tilt < cur_tilt:
            vel = -v
        else:
            vel = v
        dt = abs(abs(desired_tilt - cur_tilt) / vel)
        total_time_head += dt
        _add_point(goal, [desired_pan, desired_tilt], total_time_head)
        goal.trajectory.joint_names = ['pan_joint','tilt_joint']
        goal.goal_time_tolerance = rospy.Time(0.1)
        return goal

    @staticmethod
    def wait_for_head():
        head_topic="/movo/head_controller/state"
        msg = rospy.wait_for_message(head_topic, JointTrajectoryControllerState, timeout=15)
        assert msg.joint_names[0] == 'pan_joint', "Joint is not head joints (need pan or tilt)."
        cur_pan = msg.actual.positions[0]
        cur_tilt = msg.actual.positions[1]
        return cur_pan, cur_tilt
