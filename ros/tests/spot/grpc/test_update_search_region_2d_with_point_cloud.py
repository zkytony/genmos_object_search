# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m genmos_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_update_search_region_2d_with_point_cloud.py'
#
# Requires both point cloud and waypoints
import numpy as np
import rospy
import message_filters

from sensor_msgs.msg import PointCloud2
from rbd_spot_perception.msg import GraphNavWaypointArray

from genmos_ros.ros_utils import WaitForMessages
from genmos_ros import ros_utils
from genmos_object_search.grpc.utils import proto_utils
from genmos_object_search.grpc.common_pb2 import Pose2D
from genmos_object_search.grpc.client import GenMOSClient
from config_test_MosAgentBasic2D import TEST_CONFIG

POINT_CLOUD_TOPIC = "/graphnav_map_publisher/graphnav_points"
WAYPOINT_TOPIC = "/graphnav_waypoints"


def waypoints_msg_to_arr(waypoints_msg):
    """converts a GraphNavWaypointArray message into a numpy array"""
    arr = np.array([[wp_msg.pose_sf.position.x,
                     wp_msg.pose_sf.position.y,
                     wp_msg.pose_sf.position.z]
                    for wp_msg in waypoints_msg.waypoints])
    return arr

class UpdateSearchRegion2DTestCase:
    def __init__(self, robot_id="robot0",
                 node_name="test_update_search_region_2d_with_point_cloud",
                 world_frame="graphnav_map", debug=True, num_updates=3):
        self.node_name = node_name
        self.robot_id = robot_id
        self.world_frame = world_frame  # fixed frame of the world
        self.debug = debug  # whether to show open3d debug window
        self.num_updates = num_updates
        self._setup()

    def _setup(self):
        rospy.init_node(self.node_name)
        self._update_count = 0
        self._genmos_client = GenMOSClient()

    def run(self):
        self._genmos_client.createAgent(config=TEST_CONFIG,
                                       robot_id=self.robot_id,
                                       header=proto_utils.make_header())
        for i in range(self.num_updates):
            cloud_msg, waypoints_msg = WaitForMessages(
                [POINT_CLOUD_TOPIC, WAYPOINT_TOPIC],
                [PointCloud2, GraphNavWaypointArray],
                delay=5, verbose=True).messages
            self._update_search_region(cloud_msg, waypoints_msg)

    def _update_search_region(self, cloud_msg, waypoints_msg):
        """The first time a new search region should be created;
        Subsequent calls should update the search region"""
        # convert PointCloud2 to point cloud protobuf
        self._update_count += 1
        print(f"Received messages! Call count: {self._update_count}")

        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(cloud_msg)
        waypoints_array = waypoints_msg_to_arr(waypoints_msg)

        if self._update_count > len(waypoints_array):
            print("We have exhausted waypoints. Test complete. Please quit with Ctrl-C.")

        else:
            # Use a waypoint as the robot pose
            robot_pose = (waypoints_array[self._update_count][0],
                          waypoints_array[self._update_count][1], 0.0)
            robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)

            if rospy.get_param('map_name') == "cit_first_floor":
                layout_cut = 1.5
                region_size = -1
                brush_size = 0.5
            else:
                layout_cut = 0.6
                region_size = -1
                brush_size = 0.5

            self._genmos_client.updateSearchRegion(
                header=cloud_pb.header,
                robot_id=self.robot_id,
                is_3d=False,
                robot_pose=robot_pose_pb,
                point_cloud=cloud_pb,
                search_region_params_2d={"layout_cut": layout_cut,
                                         "region_size": region_size,
                                         "brush_size": brush_size,
                                         "debug": self.debug})

if __name__ == "__main__":
    UpdateSearchRegion2DTestCase().run()
