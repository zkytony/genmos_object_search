#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_mos_ros import ros_utils
from genmos_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("local_search_center_publisher")
    x = 1.045233416557312
    y = 1.10294508934021
    z = 0.25
    qx = 0.0
    qy = 0.0
    qz = 0.7037513720596872
    qw = 0.7104463430295829
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
