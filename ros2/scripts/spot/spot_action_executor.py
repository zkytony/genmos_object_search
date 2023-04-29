#!/usr/bin/env python
#
# To test this program
#
# python spot_action_executor.py Gosu 1.0 0.0 0.0
# python spot_action_executor.py Gosu 1.0 0.0 0.0 0 15 0
import argparse
import rclpy
import genmos_object_search.utils.math as math_utils
import spot_driver.conversions as conv
from bdai.utilities.math_helpers import SE2Pose, SE3Pose, Quaternion
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spot_utilities.spot_basic import SpotBasic

from spot_msgs.action import RobotCommand
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY


def main():
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    parser.add_argument("body_t_goal", type=float, nargs="+",
                        help="target SE3 pose in body frame."
                        "Could be either of length 3 (x,y,z), 6 (x,y,z,roll,pitch,yaw), or 7 (x,y,z,qx,qy,qz,qw)."\
                        "Note that roll, pitch, yaw are in degrees.")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node("test_spot_body")

    spot_basic = SpotBasic(robot_name=args.robot, parent_node_name=node.get_name())
    spot_basic.initialize_robot()

    robot_command_client = ActionClientWrapper(RobotCommand, 'robot_command', 'robot_node', namespace=args.robot)
    tf_listener = TFListenerWrapper('action_exec_tf',
                                    wait_for_transform=[spot_basic.body_frame_name,
                                                        spot_basic.vision_frame_name])
    if len(args.body_t_goal) == 3:
        x, y, z = args.body_t_goal
        thx, thy, thz = 0.0, 0.0, 0.0  # thx:roll, thy:pitch, thz:yaw
    elif len(args.body_t_goal) == 6:
        x, y, z, thx, thy, thz = args.body_t_goal
        thx, thy, thz = map(math_utils.to_rad, [thx, thy, thz])
    elif len(args.body_t_goal) == 7:
        x, y, z, qx, qy, qz, qw = args.body_t_goal
        thx, thy, thz = map(math_utils.to_rad, math_utils.quat_to_euler(qx, qy, qz, qw))
    else:
        raise ValueError(f"invalid body_t_goal: {args.body_t_goal}")

    # first  move the body to the goal position
    body_t_goal = SE2Pose(x, y, thz)
    vision_t_body = tf_listener.lookup_a_tform_b_se2(
        spot_basic.vision_frame_name, spot_basic.body_frame_name)
    vision_t_goal = vision_t_body * body_t_goal
    spot_basic.walk_to(vision_t_goal)

    # then tilt the body to the goal orientation
    proto_goal = RobotCommandBuilder.synchro_stand_command(
        footprint_R_body=EulerZXY(yaw=0, pitch=thy, roll=thx))
    action_goal = RobotCommand.Goal()
    conv.convert_proto_to_bosdyn_msgs_robot_command(proto_goal, action_goal.command)
    robot_command_client.send_goal_and_wait(action_goal)

    rclpy.spin_once(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
