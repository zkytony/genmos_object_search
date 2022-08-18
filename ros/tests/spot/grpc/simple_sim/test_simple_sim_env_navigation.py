import rospy
from sloop_object_search_ros.msg import KeyValAction

def main():
    rospy.init_node("test_simple_sim_env_navigation")
    action_pub = rospy.Publisher("/simple_sim_env/pomdp_action", KeyValAction, queue_size=10)
    nav_action = KeyValAction(stamp=rospy.Time.Now(),
                              type="nav",
                              keys=["goal_id", "goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"],
                              values=[100, 2, 0, 0, 0, 0, 0, 0, 1])
    action_pub.publish(nav_action)
    rospy.spin()
    # nav_action = KeyValAction(stamp=rospy.Time.Now(),
    #                           type="nav",
    #                           keys=["goal_id", "goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"],
    #                           values=[100, 2, 0, 0, 0, 0, 0, 0, 1])
    # pass

if __name__ == "__main__":
    main()
