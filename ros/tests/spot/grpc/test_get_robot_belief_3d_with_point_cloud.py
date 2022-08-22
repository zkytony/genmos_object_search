# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_get_robot_belief_3d_with_point_cloud.py'
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
from sloop_object_search.utils.misc import hash16

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class GetRobotBeliefTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)

    def run(self):
        super().run()
        response = self._sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got robot belief")

        robot_pose = proto_utils.robot_pose_from_proto(response.robot_belief.pose)

        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        marker = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id, robot_pose, header=header, scale=Vector3(x=1.2, y=0.2, z=0.2),
            lifetime=0)  # forever
        self._robot_markers_pub.publish(MarkerArray([marker]))
        print(f"Visualized robot pose")
        print(f"Check it out in rviz: roslaunch rbd_spot_perception view_graphnav_point_cloud.launch")
        print(f"Note: you may need to add the robot pose topic")
        rospy.spin()

if __name__ == "__main__":
    GetRobotBeliefTestCase(node_name="test_get_robot_belief_3d_with_point_cloud",
                           debug=False).run()
