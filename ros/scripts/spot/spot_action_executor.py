#!/usr/bin/env python

import rospy
import sys
from genmos_spot.action_executor import SpotGenMOSActionExecutor

def main():
    rospy.init_node("spot_genmos_action_executor")
    print(f"initialized node {rospy.get_name()}")
    s = SpotGenMOSActionExecutor()
    s.setup()
    rospy.spin()

if __name__ == "__main__":
    main()
