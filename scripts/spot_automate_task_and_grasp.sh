#!/bin/bash
roslaunch sloop_ros sloop_mos_spot_automate_task.launch max_steps:=50
roscd sloop_ros/scripts
python spot_arm_grasp.py -i hand_color_image $SPOT_IP
