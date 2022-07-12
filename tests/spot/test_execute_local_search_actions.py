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


def test():
    rospy.init_node("test_execute_local_search_action")
    grid_map_msg = rospy.Subscriber("/graphnav_gridmap", GridMap2d, grid_map_cb)
    pub = rospy.Publisher("/run_sloop_mos_spot/action", KeyValAction, queue_size=10)
    while GRID_MAP is None:
        print("Waiting for grid map")
        time.sleep(0.5)
    print("Got grid map")
    # Assume that the bridge is running.
    movemens = PolicyModelBasic2D.all_movements()
    action = movemens["TurnLeft"]
    robot_pose_after_action = RobotTransBasic2D.transform_pose((0, 0, 0.0), action)
    goal_pos = robot_pose_after_action[:2]
    metric_pos = GRID_MAP.to_metric_pos(*goal_pos)
    goal_yaw = to_rad(robot_pose_after_action[2])
    action_msg = KeyValAction(stamp=rospy.Time.now(),
                              type="move_2d",
                              keys=["goal_x", "goal_y", "goal_yaw", "name"],
                              values=[str(metric_pos[0]), str(metric_pos[1]), str(goal_yaw), action.name])
    pub.publish(action_msg)
    rospy.spin()

if __name__ == "__main__":
    test()
