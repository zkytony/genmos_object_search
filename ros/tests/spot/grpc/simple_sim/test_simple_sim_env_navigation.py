#!/usr/bin/env python
# 1. To run the simple_sim_env, roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=<map_name>
# 2. To get map point cloud, roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>
# 3. For rviz visualization, roslaunch sloop_object_search_ros view_simple_sim.launch
# 4. Run this test: rosrun sloop_object_search_ros test_simple_sim_env_navigation.py
import rospy
from sloop_object_search_ros.msg import KeyValAction
from sloop_object_search.utils.math import euler_to_quat

def make_nav_action(pos, orien, goal_id=100):
    goal_keys = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
    goal_values = [*pos, *orien]
    nav_action = KeyValAction(stamp=rospy.Time.now(),
                              type="nav",
                              keys=["goal_id"] + goal_keys,
                              values=list(map(str, [goal_id] + goal_values)))
    return nav_action

def test():
    rospy.init_node("test_simple_sim_env_navigation")
    action_pub = rospy.Publisher("/simple_sim_env/pomdp_action", KeyValAction, queue_size=10, latch=True)

    # Set a pose for testing
    goal_pos = (2, 0, 3)
    goal_orien = euler_to_quat(90, 90, 0)
    nav_action = make_nav_action(goal_pos, goal_orien)
    action_pub.publish(nav_action)
    print("published action:")
    print(nav_action)
    rospy.spin()

if __name__ == "__main__":
    test()
