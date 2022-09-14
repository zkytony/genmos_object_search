#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_mos_ros import ros_utils
from sloop_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("local_search_center_publisher")
    x = -0.17396485805511475
    y = -0.1354684829711914
    z = 0.5
    qx = 0.0
    qy = 0.0
    qz = 0.9992697136539744
    qw = 0.038210461576692244
    world_frame = "graphnav_map"
    pose_pub = rospy.Publisher("/local_region_center", PoseStamped, queue_size=10, latch=True)
    rate = 10
    print("Publishing local region center")
    while not rospy.is_shutdown():
        msg = ros_utils.pose_tuple_to_pose_stamped(
            (x,y,z,qx,qy,qz,qw), world_frame)
        pose_pub.publish(msg)

if __name__ == "__main__":
    main()
