#!/usr/bin/env python

import rospy
import sys
from sloop_mos_movo.action_executor import MovoSloopActionExecutor

def main():
    rospy.init_node("movo_sloop_action_executor")
    print(f"initialized node {rospy.get_name()}")
    s = MovoSloopActionExecutor()
    s.setup()
    rospy.spin()

if __name__ == "__main__":
    main()
