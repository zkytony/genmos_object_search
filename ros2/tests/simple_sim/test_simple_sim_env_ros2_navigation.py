#!/usr/bin/env python
# 0. Run config_simple_sim_lab121_lidar.py to generate the .yaml configuration file
# 1. ros2 launch spot_funcs graphnav_map_publisher.launch map_name:=lab121_lidar
# 2. ros2 launch genmos_object_search_ros2 simple_sim_env_ros2.launch map_name:=lab121_lidar
# 3. ros2 run genmos_object_search_ros2 view_simple_sim_ros2.sh
# 4. ros2 run genmos_object_search_ros2 test_simple_sim_env_ros2_navigation.py
import rclpy
import threading
import std_msgs.msg as std_msgs
from genmos_ros2 import ros2_utils
from genmos_object_search_ros2.msg import KeyValAction
from genmos_object_search.utils.math import euler_to_quat

def make_nav_action(pos, orien, goal_id=100, node=None):
    goal_keys = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
    goal_values = [*pos, *orien]
    nav_action = KeyValAction(stamp=node.get_clock().now().to_msg(),
                              type="nav",
                              keys=["action_id"] + goal_keys,
                              values=list(map(str, [goal_id] + goal_values)))
    return nav_action

ACTION_DONE_TOPIC = "/simple_sim_env/action_done"
ACTION_TOPIC = "/simple_sim_env/pomdp_action"

def test():
    rclpy.init()
    node = rclpy.create_node("test_simple_sim_env_navigation")
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, args=(), daemon=False)
    thread.start()

    action_pub = node.create_publisher(
        KeyValAction,
        ACTION_TOPIC,
        ros2_utils.latch(depth=10))

    # Set a pose for testing
    goal_pos = (2, 0, 3)
    goal_orien = euler_to_quat(90, 90, 0)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav1", node=node)
    action_pub.publish(nav_action)

    msg = ros2_utils.wait_for_messages([ACTION_DONE_TOPIC], [std_msgs.String],
                                       allow_headerless=True, verbose=True,
                                       latched_topics={ACTION_DONE_TOPIC})[0]
    print(msg.data)

    goal_pos = (4, 2, 2)
    goal_orien = euler_to_quat(0, 90, 0)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav2")
    action_pub.publish(nav_action)

    msg = ros2_utils.wait_for_messages([ACTION_DONE_TOPIC], [std_msgs.String],
                                       allow_headerless=True, verbose=True,
                                       latched_topics={ACTION_DONE_TOPIC})[0]
    print(msg.data)

    goal_pos = (-2, 4, 1)
    goal_orien = euler_to_quat(0, 90, 90)
    nav_action = make_nav_action(goal_pos, goal_orien, goal_id="nav3")
    action_pub.publish(nav_action)

    msg = ros2_utils.wait_for_messages([ACTION_DONE_TOPIC], [std_msgs.String],
                                       allow_headerless=True, verbose=True,
                                       latched_topics={ACTION_DONE_TOPIC})[0]
    print(msg.data)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test()
