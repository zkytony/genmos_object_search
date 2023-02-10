# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m genmos_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_get_robot_belief_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch genmos_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import rospy
import numpy as np

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from tf2_ros import TransformBroadcaster

from genmos_ros import ros_utils
from genmos_object_search.grpc.utils import proto_utils
from genmos_object_search.grpc.common_pb2 import Status, Voxel3D
from genmos_object_search.utils.misc import hash16

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class GetRobotBeliefTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)
        self.br = TransformBroadcaster()

    def run(self):
        super().run()
        response = self._genmos_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got robot belief")

        print(f"Visualizing robot pose")
        print(f"Check it out in rviz: roslaunch rbd_spot_perception view_graphnav_point_cloud.launch")
        print(f"Note: you may need to add the robot pose topic")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            marker, trobot = ros_utils.viz_msgs_for_robot_pose_proto(
                response.robot_belief.pose, self.world_frame, self.robot_id)
            self.br.sendTransform(trobot)
            self._robot_markers_pub.publish(MarkerArray([marker]))
            rate.sleep()

if __name__ == "__main__":
    GetRobotBeliefTestCase(node_name="test_get_robot_belief_3d_with_point_cloud",
                           debug=False).run()
