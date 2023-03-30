#!/usr/bin/env python
import rclpy
import threading

from geometry_msgs.msg import PoseStamped
from genmos_ros2 import ros2_utils
from genmos_object_search.utils.math import euler_to_quat

class SimpleSimEnvInitPosePublisher(ros2_utils.WrappedNode):
    def __init__(self):
        super().__init__("simple_sim_init_robot_pose_publisher",
                         params=[("x", 0.0),
                                 ("y", 0.0),
                                 ("z", 0.0),
                                 ("thx", 0.0),
                                 ("thy", 0.0),
                                 ("thz", 0.0),
                                 ("world_frame", "graphnav_map")])
        self.publisher = self.create_publisher(PoseStamped, "/simple_sim_env/init_robot_pose",
                                               qos_profile=ros2_utils.latch(depth=10))
        x = self.get_parameter("x").value
        y = self.get_parameter("y").value
        z = self.get_parameter("z").value
        thx = self.get_parameter("thx").value
        thy = self.get_parameter("thy").value
        thz = self.get_parameter("thz").value
        qx, qy, qz, qw = euler_to_quat(thx, thy, thz)

        world_frame = self.get_parameter("world_frame").value
        msg = ros2_utils.pose_tuple_to_pose_stamped(
            (x,y,z,qx,qy,qz,qw), world_frame, node=self)

        self.publisher.publish(msg)
        rate = self.create_rate(2)
        while rclpy.ok():
            rate.sleep()


def main():
    rclpy.init()
    node = SimpleSimEnvInitPosePublisher()
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()
    node.destroy_node()
    rclpy.shutdown()

    # pose_pub = rospy.Publisher("/simple_sim_env/init_robot_pose",
    #                            PoseStamped, queue_size=10, latch=True)

    # msg = ros2_utils.pose_tuple_to_pose_stamped(
    #     (x,y,z,qx,qy,qz,qw), world_frame, node=node)
    # pose_pub.publish(msg)
    # print("Published initial pose")
    # rospy.Timer(rospy.Duration(1/10),
    #             lambda event: pose_pub.publish(ros_utils.pose_tuple_to_pose_stamped(
    #                 (x,y,z,qx,qy,qz,qw), world_frame)))
    # rospy.spin()

if __name__ == "__main__":
    main()
