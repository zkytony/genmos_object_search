# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_process_robot_pose_observation_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints

import rospy
import numpy as np
import time
import pickle
import json

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from tf2_ros import TransformBroadcaster

from sloop_mos_ros import ros_utils
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc.common_pb2\
    import Status, Voxel3D, Box3D, Pose3D, Vec3
from sloop_object_search.grpc.observation_pb2\
    import ObjectDetectionArray, Detection3D
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import euler_to_quat, quat_to_euler
from sloop_object_search.utils.open3d_utils import draw_octree_dist
from sloop_object_search.utils.colors import lighter
from sloop_object_search.oopomdp.models.octree_belief import plot_octree_belief

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class ProcessRobotPoseObservationTestCase(CreateAgentTestCase):
    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)
        self.br = TransformBroadcaster()

    def get_and_visualize_robot_pose(self):
        response = self._sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        rospy.loginfo("got robot belief")
        marker, trobot = ros_utils.viz_msgs_for_robot_pose_proto(
            response.robot_belief.pose, self.world_frame, self.robot_id)
        self.br.sendTransform(trobot)
        self._robot_markers_pub.publish(MarkerArray([marker]))
        return response.robot_belief.pose

    def test_one_round(self, dpose):
        dx, dy, dz, dthx, dthy, dthz = dpose
        # First get robot belief and
        robot_pose_pb = self.get_and_visualize_robot_pose()
        robot_pose = proto_utils.robot_pose_from_proto(robot_pose_pb)

        print("current pose:", (*robot_pose[:3], quat_to_euler(*robot_pose[3:])))
        x,y,z,qx,qy,qz,qw = robot_pose  # This pose should be world frame
        thx, thy, thz = quat_to_euler(qx,qy,qz,qw)
        new_robot_pose = (x + dx, y + dy, z + dz, *euler_to_quat(thx + dthx, thy + dthy, thz + dthz))
        new_robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(new_robot_pose)
        print("new pose:", (*new_robot_pose[:3], quat_to_euler(*new_robot_pose[3:])))

        response = self._sloop_client.processObservation(
            self.robot_id, None, new_robot_pose_pb, frame_id=self.world_frame)
        assert response.status == Status.SUCCESSFUL

        # Get and visualize robot pose again.
        robot_pose_pb = self.get_and_visualize_robot_pose()
        robot_pose = proto_utils.robot_pose_from_proto(robot_pose_pb)
        print("got pose:", (*robot_pose[:3], quat_to_euler(*robot_pose[3:])))


    def run(self):
        super().run()
        # Let's get current robot pose, and update it.  We expect to see, after
        # update, when we request robot pose again, it is updated.
        print("----------")
        self.test_one_round((0, 0, 0, 0, 0, 0))
        time.sleep(2)
        print("----------")
        self.test_one_round((5, 0, -2, 0, 0, 90))  # rotate around world frame z axis, cw
        time.sleep(2)
        print("----------")
        self.test_one_round((0, 0, 0, 0, 0, 90))  # rotate around world frame z axis, cw

        rospy.spin()

if __name__ == "__main__":
    ProcessRobotPoseObservationTestCase(
        node_name="test_get_robot_belief_3d_with_point_cloud",
        debug=False).run()
