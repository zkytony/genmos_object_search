#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_mos_ros import ros_utils

def main():
    rospy.init_node("simple_sim_init_robot_pose_publisher")
    x = rospy.get_param("~x")
    y = rospy.get_param("~y")
    z = rospy.get_param("~z")
    qx = rospy.get_param("~qx")
    qy = rospy.get_param("~qy")
    qz = rospy.get_param("~qz")
    qw = rospy.get_param("~qw")
    world_frame = rospy.get_param("~world_frame")

    pose_pub = rospy.Publisher("/simple_sim_env/init_robot_pose",
                               PoseStamped, queue_size=10, latch=True)

    msg = ros_utils.pose_tuple_to_pose_stamped(
        (x,y,z,qx,qy,qz,qw), world_frame)
    pose_pub.publish(msg)
    print("Published initial pose")
    print(msg)
    rospy.spin()

    # To publish repeatedly
    # rospy.Timer(rospy.Duration(1/10),
    #             lambda event: pose_pub.publish(ros_utils.pose_tuple_to_pose_stamped(
    #                 (x,y,z,qx,qy,qz,qw), world_frame)))

if __name__ == "__main__":
    main()
