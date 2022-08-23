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
import time
import pickle

from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from tf2_ros import TransformBroadcaster

from sloop_mos_ros import ros_utils
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc.common_pb2 import Status, Voxel3D
from sloop_object_search.grpc.observation_pb2\
    import ObjectDetectionArray, Detection3D
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import euler_to_quat, quat_to_euler
from sloop_object_search.oopomdp.models.octree_belief import plot_octree_belief

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase

import matplotlib.pyplot as plt
def _test_visualize(octree_belief):
    fig = plt.gcf()
    ax = fig.add_subplot(1,1,1,projection="3d")
    m = plot_octree_belief(ax, octree_belief,
                           alpha="clarity", edgecolor="black", linewidth=0.1)
    ax.set_xlim([0, octree_belief.octree.dimensions[0]])
    ax.set_ylim([0, octree_belief.octree.dimensions[1]])
    ax.set_zlim([0, octree_belief.octree.dimensions[2]])
    ax.grid(False)
    fig.colorbar(m, ax=ax)
    plt.show(block=False)
    plt.pause(1)
    ax.clear()


class ProcessDetectionObservationTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)
        self.br = TransformBroadcaster()

    def get_and_visualize_belief(self):
        response = self._sloop_client.getObjectBeliefs(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got belief")

        # visualize the belief
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            bobj = pickle.loads(bobj_pb.dist_obj)
            # _test_visualize(bobj)
            msg = ros_utils.make_octree_belief_proto_markers_msg(
                bobj_pb, header, alpha_scaling=1.0)
            self._octbelief_markers_pub.publish(msg)

        print("belief visualized")

    def get_and_visualize_robot_pose(self):
        response = self._sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        print("got robot belief")
        robot_pose = proto_utils.robot_pose_from_proto(response.robot_belief.pose)
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.robot_id)

        # The camera by default looks at -z; Because in ROS, 0 degree
        # means looking at +x, therefore we rotate the marker for robot pose
        # so that it starts out looking at -z.
        marker = ros_utils.make_viz_marker_from_robot_pose_3d(
            self.robot_id, (0,0,0,*euler_to_quat(0, 90, 0)), header=header, scale=Vector3(x=1.2, y=0.2, z=0.2),
            lifetime=0)  # forever
        self._robot_markers_pub.publish(MarkerArray([marker]))

        # publish tf
        rot = np.array(quat_to_euler(*robot_pose[3:]))
        robot_pose_aligned = (*robot_pose[:3], *euler_to_quat(*rot))
        trobot = ros_utils.tf2msg_from_robot_pose(
            robot_pose_aligned, self.world_frame, self.robot_id)
        self.br.sendTransform(trobot)

        return response.robot_belief.pose


    def run(self):
        super().run()

        self.get_and_visualize_belief()
        robot_pose_pb = self.get_and_visualize_robot_pose()

        time.sleep(2)
        # First, suppose the robot receives no detection
        header = proto_utils.make_header(self.world_frame)
        object_detection = ObjectDetectionArray(header=header,
                                                robot_id=self.robot_id,
                                                detections=[])
        response = self._sloop_client.processObservation(
            self.robot_id, object_detection, robot_pose_pb, header=header)
        assert response.status == Status.SUCCESSFUL
        print("no-detection processing successful")

        # see belief now
        self.get_and_visualize_belief()
        self.get_and_visualize_robot_pose()

        rospy.spin()

if __name__ == "__main__":
    ProcessDetectionObservationTestCase(
        node_name="test_get_robot_belief_3d_with_point_cloud",
        debug=False).run()
