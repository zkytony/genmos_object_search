import rospy
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
from visualization_msgs.msg import Marker, MarkerArray
from sloop_object_search_ros.msg import KeyValAction, KeyValObservation
from sloop_mos_ros import ros_utils
from rbd_spot_perception.msg import GraphNavWaypointArray

REGION_POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
ROBOT_POSE_TOPIC = "/simple_sim_env/robot_pose"
ACTION_TOPIC = "/simple_sim_env/pomdp_action"
ACTION_DONE_TOPIC = "/simple_sim_env/action_done"
OBSERVATION_TOPIC = "/simple_sim_env/pomdp_observation"
STATE_MARKERS_TOPIC = "/simple_sim_env/state_markers"
GRAPHNAV_WAYPOINTS = "/graphnav_waypoints"

WORLD_FRAME = "graphnav_map"

SEARCH_SPACE_RESOLUTION = 0.15

def main():
    rospy.init_node("test_simple_env_messages")
    waypoints_msg, pose_stamped_msg = ros_utils.WaitForMessages(
        [GRAPHNAV_WAYPOINTS, ROBOT_POSE_TOPIC],
        [GraphNavWaypointArray, geometry_msgs.PoseStamped],
        delay=1000, verbose=True).messages
    waypoints_msg, pose_stamped_msg = ros_utils.WaitForMessages(
        [GRAPHNAV_WAYPOINTS, ROBOT_POSE_TOPIC],
        [GraphNavWaypointArray, geometry_msgs.PoseStamped],
        delay=1000, verbose=True).messages
    print(waypoints_msg.header.stamp.to_sec() - pose_stamped_msg.header.stamp.to_sec())


if __name__ == "__main__":
    main()
