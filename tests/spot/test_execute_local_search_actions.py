import rospy
import time
import rbd_spot
from sloop_object_search.oopomdp.models.policy_model import PolicyModelBasic2D
from sloop_ros.msg import GridMap2d, KeyValAction
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic2D
from sloop_object_search.utils.math import to_rad

GRID_MAP = None
def grid_map_cb(grid_map_msg):
    global GRID_MAP
    GRID_MAP = ros_msg_to_grid_map(grid_map_msg)


def test_movement_action(pub, movement_name, wait=5):
    movements = PolicyModelBasic2D.all_movements()
    action = movements[movement_name]
    robot_pose_after_action = RobotTransBasic2D.transform_pose((0, 0, 0.0), action)

    # note that by default the gripper is a bit forward with respect to the body origin
    metric_pos_x = 0.65 + robot_pose_after_action[0] * GRID_MAP.grid_size
    metric_pos_y = robot_pose_after_action[1] * GRID_MAP.grid_size
    metric_yaw = to_rad(-robot_pose_after_action[2])  # For spot's frame, we needed to reverse the angle
    action_msg = KeyValAction(stamp=rospy.Time.now(),
                              type="move_2d",
                              keys=["goal_x", "goal_y", "goal_yaw", "name"],
                              values=[str(metric_pos_x), str(metric_pos_y), str(metric_yaw), action.name])
    pub.publish(action_msg)
    time.sleep(wait)

def stow_arm(pub, wait=5):
    action_msg = KeyValAction(stamp=rospy.Time.now(),
                              type="stow_arm")
    pub.publish(action_msg)
    time.sleep(5)


def test():
    rospy.init_node("test_execute_local_search_action")
    grid_map_msg = rospy.Subscriber("/graphnav_gridmap", GridMap2d, grid_map_cb)
    pub = rospy.Publisher("/run_sloop_mos_spot/action", KeyValAction, queue_size=10)
    while GRID_MAP is None:
        print("Waiting for grid map")
        time.sleep(0.5)
    print("Got grid map")
    # Assume that the bridge is running.
    test_movement_action(pub, "TurnLeft")
    stow_arm(pub)
    test_movement_action(pub, "TurnRight")
    stow_arm(pub)
    test_movement_action(pub, "Forward")
    stow_arm(pub)


if __name__ == "__main__":
    test()
