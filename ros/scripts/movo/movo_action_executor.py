#!/usr/bin/env python

import rospy
import sys
from genmos_movo.action_executor import MovoGenMOSActionExecutor

def main():
    rospy.init_node("movo_genmos_action_executor")
    print(f"initialized node {rospy.get_name()}")
    s = MovoGenMOSActionExecutor()
    s.setup()
    rospy.spin()

if __name__ == "__main__":
    main()
