#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs.msg as geometry_msgs

import sys
import diagnostic_msgs
from pomdp_py.utils import typ

from sloop_object_search_ros.msg import KeyValAction
from sloop_mos_ros.action_executor import ActionExecutor
from visualization_msgs.msg import Marker, MarkerArray
from sloop_mos_ros import ros_utils


class MovoSloopActionExecutor(ActionExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._goal_viz_pub = rospy.Publisher(
            "~goal_markers", MarkerArray, queue_size=10, latch=True)

    def _execute_action_cb(self, msg):
        import pdb; pdb.set_trace()
