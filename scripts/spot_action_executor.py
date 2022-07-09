#!/usr/bin/env python

import rospy
from sloop_mos_spot.action_executor import SpotSloopActionExecutor

def main():
    rospy.init_node("spot_sloop_action_executor")
    print(f"initialized node {rospy.get_name()}")
    s = SpotSloopActionExecutor()
    s.setup()
    rospy.spin()

if __name__ == "__main__":
    main()
