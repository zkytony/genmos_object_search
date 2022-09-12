#!/usr/bin/env python
#
# To run a test:
# 1. run in a terminal 'roslaunch sloop_object_search_ros sloop_mos_spot_system.launch map_name:=<map_name>'
# 2. run in a terminal 'python -m sloop_object_search.grpc.server'
# 3. run in a terminal, "roslaunch sloop_object_search_ros sloop_mos_spot_exp1_local_lab121.launch"
#    (replace 'sloop_mos_spot_exp1_local_lab121.launch' with the experiment launch file you want)
# 4. run rviz, 'roslaunch sloop_object_search_ros view_spot_local_search.launch'
from sloop_mos_ros import SloopMosROS

def main():
    sr = SloopMosROS()
    sr.main()

if __name__ == "__main__":
    main()
