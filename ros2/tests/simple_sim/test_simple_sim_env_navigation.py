#!/usr/bin/env python
# 0. Run config_simple_sim_lab121_lidar.py to generate the .yaml configuration file
# 1. To run the simple_sim_env, roslaunch genmos_object_search_ros simple_sim_env.launch map_name:=<map_name>
# 2. To get map point cloud, roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>
# 3. For rviz visualization, roslaunch genmos_object_search_ros view_simple_sim.launch
# 4. Run this test: rosrun genmos_object_search_ros test_simple_sim_env_navigation.py
import rospy
import std_msgs.msg as std_msgs
from genmos_ros import ros_utils
from genmos_object_search_ros.msg import KeyValAction
from genmos_object_search.utils.math import euler_to_quat

def make_nav_action(pos, orien, goal_id=100):
    goal_keys = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
    goal_values = [*pos, *orien]
    nav_action = KeyValAction(stamp=rospy.Time.now(),
                              type="nav",
                              keys=["goal_id"] + goal_keys,
                              values=list(map(str, [goal_id] + goal_values)))
    return nav_action

ACTION_DONE_TOPIC = "/simple_sim_env/action_done"

def test():
    rospy.init_node("test_simple_sim_env_navigation")
    action_pub = rospy.Publisher("/simple_sim_env/pomdp_action", KeyValAction, queue_size=10, latch=True)

    # Set a pose for testing
    goal_pos = (2, 0, 3)
    goal_orien = euler_to_quat(90, 90, 0)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav1")
    action_pub.publish(nav_action)

    msg = ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                    allow_headerless=True, verbose=True).messages[0]
    print(msg.data)

    goal_pos = (4, 2, 2)
    goal_orien = euler_to_quat(0, 90, 0)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav2")
    action_pub.publish(nav_action)

    msg = ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                    allow_headerless=True, verbose=True).messages[0]
    print(msg.data)

    goal_pos = (-2, 4, 1)
    goal_orien = euler_to_quat(0, 90, 90)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav3")
    action_pub.publish(nav_action)

    msg = ros_utils.WaitForMessages([ACTION_DONE_TOPIC], [std_msgs.String],
                                    allow_headerless=True, verbose=True).messages[0]
    print(msg.data)


if __name__ == "__main__":
    test()
