#!/bin/bash
roslaunch sloop_ros sloop_mos_spot_automate_task.launch max_steps:=50
cd ~/repo/robotdev/spot/spot-sdk/python/examples/arm_grasp
python arm_grasp.py -i hand_color_image $SPOT_IP
