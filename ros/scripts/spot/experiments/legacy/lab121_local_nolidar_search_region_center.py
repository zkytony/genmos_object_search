#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from genmos_ros import ros_utils
from genmos_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("local_search_center_publisher")
    x = 3.7577287769317627
    y = -1.2946953773498535
    z = 0.15
    qx = 0.0
    qy = 0.0
    qz = -0.02592851270639481
    qw = 0.9971132638563158
    world_frame = "graphnav_map"
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
