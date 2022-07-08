#!/usr/bin/env python

import rospy
import diagnostic_msgs
from sloop_ros.msg import SloopMosAction
from sloop_object_search.ros.framework import ActionExecutor

class SpotSloopActionExecutor(ActionExecutor):
    def __init__(self):
        super().__init__(action_msg_type=SloopMosAction)

    def _execute_action_cb(self, msg):
        print(msg)

    @classmethod
    def action_to_ros_msg(self, agent, action, goal_id):
        print(msg)
