# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_create_agent_3d_with_point_cloud.py'
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
from sloop_object_search.grpc.utils import proto_utils as pbutil
from sloop_object_search.grpc.common_pb2 import Pose3D, Vec3, Quaternion, Status
from sloop_object_search.grpc.client import SloopObjectSearchClient
from config_test_MosAgentBasic3D import TEST_CONFIG

from test_update_search_region_3d_with_point_cloud import UpdateSearchRegion3DTestCase as BaseTestCase3D
from test_update_search_region_3d_with_point_cloud import POINT_CLOUD_TOPIC, WAYPOINT_TOPIC, FAKE_ROBOT_POSE_TOPIC


class CreateAgentTestCase(BaseTestCase3D):
    def run(self):
        response = self._sloop_client.getAgentCreationStatus(self.robot_id)
        assert response.status == Status.FAILED

        self.config = TEST_CONFIG
        self._sloop_client.createAgent(config=TEST_CONFIG,
                                       robot_id=self.robot_id,
                                       header=pbutil.make_header())
        response = self._sloop_client.getAgentCreationStatus(self.robot_id)
        assert response.status == Status.PENDING

        cloud_msg, waypoints_msg = WaitForMessages(
            [POINT_CLOUD_TOPIC, WAYPOINT_TOPIC],
            [PointCloud2, GraphNavWaypointArray],
            delay=5, verbose=True).messages
        self._update_search_region(cloud_msg, waypoints_msg)

        print("waiting for agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        response = self._sloop_client.getAgentCreationStatus(self.robot_id)
        assert response.status == Status.SUCCESSFUL

        print("create agent test passed.")



if __name__ == "__main__":
    CreateAgentTestCase(node_name="test_create_agent_3d_with_point_cloud",
                        debug=False).run()
