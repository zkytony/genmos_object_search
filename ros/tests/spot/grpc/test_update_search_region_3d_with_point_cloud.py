# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_update_search_region_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import numpy as np
import rospy
import message_filters

import geometry_msgs.msg as geometry_msgs
from sensor_msgs.msg import PointCloud2
from rbd_spot_perception.msg import GraphNavWaypointArray

from sloop_mos_ros.ros_utils import pose_tuple_to_pose_stamped, WaitForMessages
from sloop_object_search.grpc.utils.proto_utils import make_header, pointcloud2_to_pointcloudproto
from sloop_object_search.grpc.common_pb2 import Pose3D, Vec3, Quaternion
from sloop_object_search.grpc.client import SloopObjectSearchClient
from config_test_MosAgentBasic3D import TEST_CONFIG

POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
WAYPOINT_TOPIC = "/graphnav_waypoints"
FAKE_ROBOT_POSE_TOPIC = "/fake_robot_pose"


def waypoints_msg_to_arr(waypoints_msg):
    """converts a GraphNavWaypointArray message into a numpy array"""
    arr = np.array([[wp_msg.pose_sf.position.x,
                     wp_msg.pose_sf.position.y,
                     wp_msg.pose_sf.position.z]
                    for wp_msg in waypoints_msg.waypoints])
    return arr

class UpdateSearchRegion3DTestCase:
    def __init__(self, robot_id="robot0",
                 node_name="test_update_search_region_3d_with_point_cloud",
                 world_frame="graphnav_map", debug=True, num_updates=1):
        self.node_name = node_name
        self.world_frame = world_frame
        self.debug = debug
        self.robot_id = robot_id
        self.num_updates = num_updates
        self._setup()

    def _setup(self):
        rospy.init_node(self.node_name)
        self.wyp_sub = rospy.Subscriber(WAYPOINT_TOPIC, GraphNavWaypointArray, self._waypoint_cb)
        self.robot_pose_pub = rospy.Publisher(
            FAKE_ROBOT_POSE_TOPIC, geometry_msgs.PoseStamped, queue_size=10)
        self._update_count = 0

        self._sloop_client = SloopObjectSearchClient()

    def run(self):
        self._sloop_client.createAgent(config=TEST_CONFIG,
                                       robot_id=self.robot_id,
                                       header=make_header())
        for i in range(self.num_updates):
            cloud_msg, waypoints_msg = WaitForMessages(
                [POINT_CLOUD_TOPIC, WAYPOINT_TOPIC],
                [PointCloud2, GraphNavWaypointArray],
                delay=5, verbose=True).messages
            self._update_search_region(cloud_msg, waypoints_msg)


    def _waypoint_cb(self, waypoints_msg):
        # Publish a fake robot pose using waypoint
        waypoints_array = waypoints_msg_to_arr(waypoints_msg)
        if self._update_count == 0:
            waypoint = waypoints_array[0]
        else:
            waypoint = waypoints_array[self._update_count-1]
        pose_stamped = pose_tuple_to_pose_stamped((*waypoint, 0, 0, 0, 1), "body")

        rate = rospy.Rate(10)
        self.robot_pose_pub.publish(pose_stamped)
        rate.sleep()

    def _update_search_region(self, cloud_msg, waypoints_msg):
        """The first time a new search region should be created;
        Subsequent calls should update the search region"""
        # convert PointCloud2 to point cloud protobuf
        self._update_count += 1
        print(f"Received messages! Call count: {self._update_count}")

        cloud_pb = pointcloud2_to_pointcloudproto(cloud_msg)
        waypoints_array = waypoints_msg_to_arr(waypoints_msg)

        if self._update_count > len(waypoints_array):
            print("We have exhausted waypoints. Test complete. Please quit with Ctrl-C.")

        else:
            # Use a waypoint as the robot pose
            robot_pose_pb = Pose3D(
                position=Vec3(x=waypoints_array[self._update_count-1][0],
                              y=waypoints_array[self._update_count-1][1],
                              z=waypoints_array[self._update_count-1][2]),
                rotation=Quaternion(x=0, y=0, z=0, w=1))
            if rospy.get_param('map_name') == "cit_first_floor":
                layout_cut = 1.5
                region_size = 12.0
                brush_size = 0.5
            else:
                layout_cut = 0.6
                region_size = 5.0
                brush_size = 0.5

            self._sloop_client.updateSearchRegion(
                header=cloud_pb.header,
                robot_id=self.robot_id,
                is_3d=True,
                robot_pose_3d=robot_pose_pb,
                point_cloud=cloud_pb,
                search_region_params_3d={"octree_size": 64,
                                         "search_space_resolution": 0.15,
                                         "debug": self.debug,
                                         "region_size_x": 4.0,
                                         "region_size_y": 4.0,
                                         "region_size_z": 2.5})

if __name__ == "__main__":
    UpdateSearchRegion3DTestCase(num_updates=10).run()
