#!/bin/bash
roslaunch sloop_object_search sloop_mos_spot_automate_task.launch max_steps:=50
cd ~/repo/robotdev/spot/ros_ws/src/sloop_object_search/scripts
python spot_arm_grasp.py -i hand_color_image $SPOT_IP
