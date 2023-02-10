#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_mos_ros import ros_utils
from genmos_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("local_search_center_publisher")
    x = 1.03092479706
    y = 0.466156184673
    z = 0.75
    qx = 0.0
    qy = 0.0
    qz = 0.207342692613
    qw = 0.978268372084
    world_frame = "map"
    pose_pub = rospy.Publisher("/local_region_center", PoseStamped, queue_size=10, latch=True)
    rate = rospy.Rate(5)
    print("Publishing local region center")
    while not rospy.is_shutdown():
        msg = ros_utils.pose_tuple_to_pose_stamped(
            (x,y,z,qx,qy,qz,qw), world_frame)
        pose_pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()
