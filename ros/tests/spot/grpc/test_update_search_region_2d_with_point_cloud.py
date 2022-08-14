# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_update_search_region_2d_with_point_cloud.py'
#
# Requires both point cloud and waypoints
import numpy as np
import rospy
import message_filters

from sensor_msgs.msg import PointCloud2
from rbd_spot_perception.msg import GraphNavWaypointArray

from sloop_object_search.grpc.utils.proto_utils import pointcloud2_to_pointcloudproto
from sloop_object_search.grpc.common_pb2 import Pose2D, BasicParam
from sloop_object_search.grpc.client import SloopObjectSearchClient
from config_test_SloopMosTopo2DAgent import TEST_CONFIG

POINT_CLOUD_TOPIC = "/graphnav_map_publisher/graphnav_points"
WAYPOINT_TOPIC = "/graphnav_waypoints"


def waypoints_msg_to_arr(waypoints_msg):
    """converts a GraphNavWaypointArray message into a numpy array"""
    arr = np.array([[wp_msg.pose_sf.position.x,
                     wp_msg.pose_sf.position.y,
                     wp_msg.pose_sf.position.z]
                    for wp_msg in waypoints_msg.waypoints])
    return arr

class TestCase:
    def __init__(self):
        rospy.init_node("test_update_search_region_2d_with_point_cloud")

        self.pcl_sub = message_filters.Subscriber(POINT_CLOUD_TOPIC, PointCloud2)
        self.wyp_sub = message_filters.Subscriber(WAYPOINT_TOPIC, GraphNavWaypointArray)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pcl_sub, self.wyp_sub], 10, 0.2)  # allow 0.2s difference
        self.ts.registerCallback(self._cloud_waypoints_callback)

        self._sloop_client = SloopObjectSearchClient()
        rospy.spin()

    def _cloud_waypoints_callback(self, cloud_msg, waypoints_msg):
        # convert PointCloud2 to point cloud protobuf
        print("Received messages!")
        cloud_pb = pointcloud2_to_pointcloudproto(cloud_msg)
        waypoints_array = waypoints_msg_to_arr(waypoints_msg)
        # Use the first waypoint as the robot pose
        robot_pose_pb = Pose2D(x=waypoints_array[0][0],
                               y=waypoints_array[0][1],
                               th=0.0)
        self._sloop_client.UpdateSearchRegion(
            robot_pose_2d=robot_pose_pb,
            point_cloud=cloud_pb,
            search_region_params_2d={"layout_cut": 0.5})

if __name__ == "__main__":
    TestCase()
