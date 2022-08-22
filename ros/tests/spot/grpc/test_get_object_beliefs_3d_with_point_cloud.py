# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_get_object_beliefs_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import rospy
import numpy as np

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3

from sloop_object_search.grpc.utils import proto_utils as pbutil
from sloop_object_search.grpc.common_pb2 import Status, Voxel3D
from sloop_object_search.utils.misc import hash16

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


def make_octnode_marker_msg(pos, res, prob, header, lifetime=1.0):
    """
    Creates an rviz marker for a OctNode, specified
    by the given 3D position (in frame of header),
    resolution (in meters), and with transparency determined by
    given probability.
    """
    marker = Marker()
    marker.header = header
    marker.id = hash16((*pos, res))
    marker.type = Marker.CUBE
    marker.pose.position = Point(x=pos[0] + res/2,
                                 y=pos[1] + res/2,
                                 z=pos[2] + res/2)
    marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
    marker.scale = Vector3(x=res, y=res, z=res)
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(lifetime)
    marker.color = ColorRGBA(r=0.0, g=0.8, b=0.0, a=prob)
    return marker


class GetObjectBeliefsTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)

    def run(self):
        super().run()
        response = self._sloop_client.getObjectBeliefs(
            self.robot_id, header=pbutil.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got belief")

        # visualize the belief
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            hist_pb = bobj_pb.dist
            for i in range(hist_pb.length):
                voxel = Voxel3D()
                hist_pb.values[i].Unpack(voxel)

                pos = [voxel.pos.x, voxel.pos.y, voxel.pos.z]
                prob = hist_pb.probs[i]
                marker = make_octnode_marker_msg(pos, voxel.res, prob, header, lifetime=0)  # 0 is forever
                markers.append(marker)
            self._octbelief_markers_pub.publish(MarkerArray(markers))
            print(f"Visualized belief for object {bobj_pb.object_id}")
            print(f"Check it out in rviz: roslaunch rbd_spot_perception view_graphnav_point_cloud.launch")
            print(f"Note: you may need to add the octree belief topic")
            break
        rospy.spin()

if __name__ == "__main__":
    GetObjectBeliefsTestCase(node_name="test_get_object_beliefs_3d_with_point_cloud",
                             debug=False).run()
