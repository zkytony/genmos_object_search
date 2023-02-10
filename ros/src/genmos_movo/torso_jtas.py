# Note: adapted code from MOS3D
# /author: Kaiyu Zheng

import sys
import rospy
import actionlib
from copy import copy

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import JointState


class TorsoJTAS(object):
    """Extension to movo's TorsoActionClient that allows
    specifying velocity of torso movement."""
    def __init__(self, timeout=10.0):
        self._client = actionlib.SimpleActionClient(
            'movo/torso_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        server_up = self._client.wait_for_server(timeout=rospy.Duration(timeout))
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
    def make_goal(desired_height, v=0.05):
        def _add_point(goal, positions, time):
            point = JointTrajectoryPoint()
            point.positions = copy(positions)
            point.velocities = [0.0] * len(goal.trajectory.joint_names)
            point.time_from_start = rospy.Duration(time)
            goal.trajectory.points.append(point)

        goal = FollowJointTrajectoryGoal()
        current_height = TorsoJTAS.wait_for_torso_height()
        total_time_torso = 0.0
        _add_point(goal, [current_height], 0.0)
        if desired_height < current_height:
            vel = -v
        else:
            vel = v
        dt = abs(abs(desired_height - current_height) / vel)
        _add_point(goal, [desired_height], total_time_torso)
        goal.goal_time_tolerance = rospy.Time(0.1)
        goal.trajectory.joint_names = ['linear_joint']
        return goal

    @staticmethod
    def wait_for_torso_height():
        torso_topic="/movo/linear_actuator/joint_states"
        msg = rospy.wait_for_message(torso_topic, JointState, timeout=15)
        assert msg.name[0] == 'linear_joint', "Joint is not linear joint (not torso)."
        position = msg.position[0]
        return position
