#!/usr/bin/env python
#
# To run a test:
# 1. run in one terminal, "roslaunch sloop_object_search_ros sloop_mos_spot_exp1_local_lab121.launch"
#    (replace 'sloop_mos_spot_exp1_local_lab121.launch' with the experiment launch file you want)
# 2. run rviz
from sloop_mos_ros import SloopMosROS

def main():
    sr = SloopMosROS()
    sr.main()

if __name__ == "__main__":
    main()
