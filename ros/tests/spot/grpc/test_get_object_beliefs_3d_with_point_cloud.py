# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m genmos_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_get_object_beliefs_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch genmos_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import rospy
import numpy as np
import pickle

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3

from genmos_ros import ros_utils
from genmos_object_search.grpc.utils import proto_utils as pbutil
from genmos_object_search.grpc.common_pb2 import Status, Voxel3D
from genmos_object_search.utils.misc import hash16
from genmos_object_search.utils.colors import cmaps
from genmos_object_search.utils.open3d_utils import draw_octree_dist

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class GetObjectBeliefsTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)

    def run(self, o3dviz=True):
        """o3dviz: visualize octree belief in open3d"""
        super().run()
        response = self._genmos_client.getObjectBeliefs(
            self.robot_id, header=pbutil.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got belief")

        # visualize the belief
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            msg = ros_utils.make_octree_belief_proto_markers_msg(bobj_pb, header)
            self._octbelief_markers_pub.publish(msg)
            print(f"Visualized belief for object {bobj_pb.object_id}")
            print(f"Check it out in rviz: roslaunch rbd_spot_perception view_graphnav_point_cloud.launch")
            print(f"Note: you may need to add the octree belief topic")
            break

        rospy.spin()


if __name__ == "__main__":
    GetObjectBeliefsTestCase(node_name="test_get_object_beliefs_3d_with_point_cloud",
                             debug=False).run()
