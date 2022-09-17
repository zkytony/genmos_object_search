#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_mos_ros import ros_utils
from sloop_object_search.utils.math import euler_to_quat

def main():
    rospy.init_node("simple_sim_init_robot_pose_publisher")
    x = rospy.get_param("~x")
    y = rospy.get_param("~y")
    z = rospy.get_param("~z")
    thx = rospy.get_param("~thx")
    thy = rospy.get_param("~thy")
    thz = rospy.get_param("~thz")
    qx, qy, qz, qw = euler_to_quat(thx, thy, thz)
    world_frame = rospy.get_param("~world_frame")

    pose_pub = rospy.Publisher("/simple_sim_env/init_robot_pose",
                               PoseStamped, queue_size=10, latch=True)

    msg = ros_utils.pose_tuple_to_pose_stamped(
        (x,y,z,qx,qy,qz,qw), world_frame)
    pose_pub.publish(msg)
    print("Published initial pose")
    rospy.Timer(rospy.Duration(1/10),
                lambda event: pose_pub.publish(ros_utils.pose_tuple_to_pose_stamped(
                    (x,y,z,qx,qy,qz,qw), world_frame)))
    rospy.spin()

if __name__ == "__main__":
    main()
