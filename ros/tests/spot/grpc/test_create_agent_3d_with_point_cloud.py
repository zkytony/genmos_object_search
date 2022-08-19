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

from sloop_mos_ros.ros_utils import pose_tuple_to_pose_stamped
from sloop_object_search.grpc.utils import proto_utils as pbutil
from sloop_object_search.grpc.common_pb2 import Pose3D, Vec3, Quaternion, Status
from sloop_object_search.grpc.client import SloopObjectSearchClient
from config_test_MosAgentBasic3D import TEST_CONFIG

from test_update_search_region_3d_with_point_cloud import UpdateSearchRegion3DTestCase as BaseTestCase3D


class CreateAgentTestCase(BaseTestCase3D):
    def run(self):
        response = self._sloop_client.getAgentCreationStatus(self.agent_name)
        assert response.status == Status.FAILED

        self._sloop_client.createAgent(config=TEST_CONFIG,
                                       agent_name=self.agent_name,
                                       header=pbutil.make_header())
        response = self._sloop_client.getAgentCreationStatus(self.agent_name)
        assert response.status == Status.PENDING

        print("waiting for agent creation...")
        self._sloop_client.waitForAgentCreation(self.agent_name)
        response = self._sloop_client.getAgentCreationStatus(self.agent_name)
        assert response.status == Status.SUCCESS

        print("test passed.")
        self._sloop_client.channel.close()



if __name__ == "__main__":
    CreateAgentTestCase(node_name="test_create_agent_3d_with_point_cloud",
                        debug=False).run()
