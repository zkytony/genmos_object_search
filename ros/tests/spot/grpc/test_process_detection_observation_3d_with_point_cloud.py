# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_process_detection_observation_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import rospy
import numpy as np

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3

from sloop_mos_ros import ros_utils
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc.common_pb2 import Status, Voxel3D
from sloop_object_search.grpc.observation_pb2\
    import ObjectDetectionArray, Detection3D
from sloop_object_search.utils.misc import hash16

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class ProcessDetectionObservationTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)

    def run(self):
        super().run()

        # First, suppose the robot receives no detection
        header = proto_utils.make_header(self.world_frame)
        object_detection = ObjectDetectionArray(header=header,
                                                robot_id=self.robot_id,
                                                detections=[])
        response = self._sloop_client.processObservation(
            self.robot_id, object_detection, header=header)
        assert response.status == Status.SUCCESSFUL
        print("no-detection processing successful")


        rospy.spin()

if __name__ == "__main__":
    ProcessDetectionObservationTestCase(
        node_name="test_get_robot_belief_3d_with_point_cloud",
        debug=False).run()
