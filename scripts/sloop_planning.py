#!/usr/bin/env python
from sloop_ros.sloop_planning import sloop_planner

if __name__ == "__main__":
    try:
        sloop_planner()
    except rospy.ROSInterruptException:
        pass
