#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from genmos_ros import ros_utils
from genmos_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("local_search_center_publisher")

    search_region_3d_config_prefix = rospy.get_param("~search_region_3d_config_prefix")
    center_pose = (
        rospy.get_param("{}/center_x".format(search_region_3d_config_prefix)),
        rospy.get_param("{}/center_y".format(search_region_3d_config_prefix)),
        rospy.get_param("{}/center_z".format(search_region_3d_config_prefix)),
        rospy.get_param("{}/center_qx".format(search_region_3d_config_prefix), 0.0),
        rospy.get_param("{}/center_qy".format(search_region_3d_config_prefix), 0.0),
        rospy.get_param("{}/center_qz".format(search_region_3d_config_prefix), 0.0),
        rospy.get_param("{}/center_qw".format(search_region_3d_config_prefix), 1.0)
    )
    world_frame = rospy.get_param("~world_frame")
    pose_pub = rospy.Publisher("/local_region_center", PoseStamped, queue_size=10, latch=True)
    rate = rospy.Rate(5)
    rospy.loginfo(f"Publishing local region center: {center_pose}")
    while not rospy.is_shutdown():
        msg = ros_utils.pose_tuple_to_pose_stamped(center_pose, world_frame)
        pose_pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()
