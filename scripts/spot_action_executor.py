#!/usr/bin/env python

import rospy
import sys
from sloop_mos_spot.action_executor import SpotSloopActionExecutor

def main():
    rospy.init_node("spot_sloop_action_executor")
    print(f"initialized node {rospy.get_name()}")
    status_topic = rospy.get_param("~status_topic")
    s = SpotSloopActionExecutor(status_topic=status_topic)
    s.setup()
    rospy.spin()

if __name__ == "__main__":
    main()
