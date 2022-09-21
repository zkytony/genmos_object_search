import rospy
import tf2_ros
import actionlib
import pomdp_py
import std_msgs.msg as std_msgs
from pomdp_py.utils import typ
from actionlib_msgs.msg import GoalStatus
from sloop_object_search_ros.msg import (PlanNextStepAction,
                                         PlanNextStepResult,
                                         KeyValAction,
                                         DefaultBelief,
                                         DefaultObservation)
from sloop_object_search.utils.misc import import_class
from . import ros_utils


class ActionExecutor:
    """ActionExecutor is meant to be run as a node by itself,
    which subscribes to the ~action topic that the SloopMosROS
    publishes when one planning step is performed.

    It:
    - subscribes to actions published at a topic
    - executes a received action;
    - publishes status as the robot executes.

    Important functions to implement:
    - execute_action_cb: called when an action message is received, and
        execute that action on the robot.
    """
    def __init__(self,
                 action_topic="~action", status_topic="~status", done_topic="~done"):
        self.node_name = rospy.get_name()
        self._action_topic = action_topic  # The topic to subscribe to to receive actions
        self._status_topic = status_topic  # The topic to publish status
        self._done_topic = done_topic  # publish when an action is done
        self._action_msg_type = KeyValAction

    @property
    def status_topic(self):
        return "{}/{}".format(self.node_name, self._status_topic)

    def setup(self):
        self._status_pub = rospy.Publisher(self._status_topic,
                                           GoalStatus,
                                           queue_size=10, latch=True)
        self._done_pub = rospy.Publisher(self._done_topic,
                                         std_msgs.String,
                                         queue_size=10, latch=True)
        self._action_sub = rospy.Subscriber(self._action_topic,
                                            self._action_msg_type,
                                            self._execute_action_cb)

    def _execute_action_cb(self, action_msg):
        """Handles action execution"""
        raise NotImplementedError

    def publish_status(self, status, text, action_id, stamp, pub_done=False):
        status = GoalStatus(status=status,
                            text=text)
        status.goal_id.id = action_id
        status.goal_id.stamp = stamp
        if status.status == GoalStatus.ABORTED or status.status == GoalStatus.REJECTED:
            rospy.logerr(text)
        else:
            rospy.loginfo(text)
        self._status_pub.publish(status)

        if pub_done:
            if status.status == GoalStatus.ABORTED or status.status == GoalStatus.REJECTED\
               or status.status == GoalStatus.SUCCEEDED:
                # We are done - whether we succeeded or not
                self._done_pub.publish(std_msgs.String(f"{action_id} done"))
