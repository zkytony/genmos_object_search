roslaunch sloop_ros sloop_mos_spot_automate_task.launch
cd ~/repo/robotdev/spot/spot-sdk/python/examples
python arm_grasp.py -i hand_color_image $SPOT_IP