# A pomdp_py.Action that can be converted to the
# ROS message sloop_ros/Action
import pomdp_py
from sloop_ros.msg import DefaultAction
from sloop_ros.utils.misc import tobeoverriden

class BaseAction(pomdp_py.SimpleAction):
    """
    The base class of all actions used in sloop_ros.
    It has a 'to_ros_msg' function that outputs a
    ROS message object corresponding to this action.
    By default, the message type is DefaultAction.
    If you would like your action to be converted to
    a different message type, you can override the
    "to_ros_msg" function.
    """
    def __init__(self, name, data=None):
        super().__init__(name)
        self.data = data

    @tobeoverriden
    def to_ros_msg(self, goal_id):
        # All Action message types are assumed to have a goal_id
        action_msg = DefaultAction()
        action_msg.goal_id = goal_id
        action_msg.name = self.name
        action_msg.data = str(self.data)
        return action_msg
