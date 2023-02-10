#!/usr/bin/env python
import rospy
import tf2_ros
import geometry_msgs, std_msgs
from genmos_ros.ros_utils import transform_to_pose_stamped

def main():
    rospy.init_node("movo_stream_camera_pose")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    pose_pub = rospy.Publisher("/movo_camera_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

    rate = rospy.Rate(10)
    last_ex = None
    message_printed = False
    while not rospy.is_shutdown():
        try:
            timestamp = rospy.Time(0)
            trans_stamped = tfBuffer.lookup_transform("map", "kinect2_color_frame", timestamp)
            pose_msg = transform_to_pose_stamped(trans_stamped.transform,
                                                 "map",
                                                 stamp=trans_stamped.header.stamp)
            pose_pub.publish(pose_msg)
            if not message_printed:
                rospy.loginfo("publishing camera pose")
                message_printed = True

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rate.sleep()
            if last_ex is None or str(last_ex) != str(ex):
                rospy.logerr(ex)
            last_ex = ex
            continue

if __name__ == "__main__":
    main()
