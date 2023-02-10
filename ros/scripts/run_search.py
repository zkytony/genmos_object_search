#!/usr/bin/env python
#
# To run a test (For Spot):
# 1. run in a terminal 'roslaunch genmos_object_search_ros sloop_mos_spot_system.launch map_name:=<map_name>'
# 2. run in a terminal 'roslaunch genmos_object_search_ros sloop_mos_spot_action_executor.launch'
# 3. run in a terminal 'python -m genmos_object_search.grpc.server'
# 4. run in a terminal, "roslaunch genmos_object_search_ros sloop_mos_spot_exp1_local_lab121.launch"
#    (replace 'sloop_mos_spot_exp1_local_lab121.launch' with the experiment launch file you want)
# 5. run rviz, 'roslaunch genmos_object_search_ros view_spot_local_search.launch'
###########################################################################
# To run a test (For MOVO):
# 1. bootup movo. Then start navigation stack on movo2: roslaunch movo_demos map_nav.launch map_file:=scili8_living_room
# 2. run in a terminal 'roslaunch genmos_object_search_ros sloop_mos_movo_system.launch map_name:=<map_name>'
# 3. run in a terminal 'python -m genmos_object_search.grpc.server'
# 4. run in a terminal, "roslaunch genmos_object_search_ros sloop_mos_movo_exp1_scili8_living_room.launch"
# 5. run rviz, 'roslaunch genmos_object_search_ros view_movo_object_search.launch'
from sloop_mos_ros import SloopMosROS

def main():
    sr = SloopMosROS()
    sr.setup()
    sr.run()

if __name__ == "__main__":
    main()
