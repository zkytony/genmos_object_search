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
from sloop_object_search.grpc.common_pb2\
    import Status, Voxel3D, Box3D, Pose3D, Vec3
from sloop_object_search.grpc.observation_pb2\
    import ObjectDetectionArray, Detection3D
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import euler_to_quat, quat_to_euler
from sloop_object_search.utils.open3d_utils import draw_octree_dist
from sloop_object_search.oopomdp.models.octree_belief import plot_octree_belief

from test_create_agent_3d_with_point_cloud import CreateAgentTestCase


class ProcessDetectionObservationTestCase(CreateAgentTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)
        self._object_markers_pub = rospy.Publisher(
            "~objects", MarkerArray, queue_size=10, latch=True)
        self.br = TransformBroadcaster()

    def get_and_visualize_belief(self, o3dviz=True):
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
            if o3dviz:
                draw_octree_dist(bobj.octree_dist)

            # clear and republish
            clear_msg = ros_utils.clear_markers(header, ns="octnode")
            self._octbelief_markers_pub.publish(clear_msg)
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
            self.robot_id, (0,0,0,*euler_to_quat(0, 90, 0)),
            header=header, scale=Vector3(x=1.2, y=0.2, z=0.2),
            lifetime=0)  # forever
        self._robot_markers_pub.publish(MarkerArray([marker]))

        # publish tf
        trobot = ros_utils.tf2msg_from_robot_pose(
            robot_pose, self.world_frame, self.robot_id)
        self.br.sendTransform(trobot)

        return response.robot_belief.pose

    def make_up_object(self, objid, object_loc, objsizes):
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="object")
        self._object_markers_pub.publish(clear_msg)
        marker = ros_utils.make_viz_marker_for_object(
            objid, (*object_loc, 0, 0, 0, 1), header,
            scale=Vector3(*objsizes), lifetime=0)
        self._object_markers_pub.publish(MarkerArray([marker]))

        tobj = ros_utils.tf2msg_from_object_loc(
            object_loc, self.world_frame, objid)
        self.br.sendTransform(tobj)

    def run(self, o3dviz=False):
        super().run()

        self.get_and_visualize_belief(o3dviz=o3dviz)
        robot_pose_pb = self.get_and_visualize_robot_pose()

        # ############### Test 1 #######################
        # # First, suppose the robot receives no detection
        # print("testing no-detection processing")
        # time.sleep(2)
        # header = proto_utils.make_header(self.world_frame)
        # object_detection = ObjectDetectionArray(header=header,
        #                                         robot_id=self.robot_id,
        #                                         detections=[])
        # response = self._sloop_client.processObservation(
        #     self.robot_id, object_detection, robot_pose_pb, header=header)
        # assert response.status == Status.SUCCESSFUL
        # print("no-detection processing done")

        # # see belief now
        # self.get_and_visualize_belief(o3dviz=o3dviz)
        # self.get_and_visualize_robot_pose()

        # ############### Test 2 #######################
        # # suppose the robot receive detection outside FOV
        # print("testing out-of-fov processing")
        # time.sleep(3)
        # robot_pose = proto_utils.robot_pose_from_proto(robot_pose_pb)
        # target_id = self.config["agent_config"]["targets"][0]
        # obj_loc = [0.87, 1.92, 0.25]
        # objsizes = [0.75, 1.0, 0.75]
        # self.make_up_object(target_id, obj_loc, objsizes)
        # objbox = Box3D(center=proto_utils.posetuple_to_poseproto((*obj_loc, 0, 0, 0, 1)),
        #                sizes=Vec3(x=objsizes[0],
        #                           y=objsizes[1],
        #                           z=objsizes[2]))

        # header = proto_utils.make_header(self.world_frame)
        # object_detection = ObjectDetectionArray(
        #     header=header,
        #     robot_id=self.robot_id,
        #     detections=[Detection3D(label=target_id,
        #                             box=objbox)])
        # response = self._sloop_client.processObservation(
        #     self.robot_id, object_detection, robot_pose_pb, header=header)
        # assert response.status == Status.SUCCESSFUL
        # print("out-of-fov detection processing successful")
        # # see belief now
        # self.get_and_visualize_belief(o3dviz=o3dviz)
        # self.get_and_visualize_robot_pose()


        ############### Test 3 #######################
        # suppose the robot receive detection within FOV
        print("testing within-fov processing")
        time.sleep(3)
        robot_pose = proto_utils.robot_pose_from_proto(robot_pose_pb)
        target_id = self.config["agent_config"]["targets"][0]
        obj_loc = [4.20, 4.39, 0.45]
        objsizes = [0.75, 1.0, 0.75]
        self.make_up_object(target_id, obj_loc, objsizes)
        objbox = Box3D(center=proto_utils.posetuple_to_poseproto((*obj_loc, 0, 0, 0, 1)),
                       sizes=Vec3(x=objsizes[0],
                                  y=objsizes[1],
                                  z=objsizes[2]))

        header = proto_utils.make_header(self.world_frame)
        object_detection = ObjectDetectionArray(
            header=header,
            robot_id=self.robot_id,
            detections=[Detection3D(label=target_id,
                                    box=objbox)])
        response = self._sloop_client.processObservation(
            self.robot_id, object_detection, robot_pose_pb, header=header)
        assert response.status == Status.SUCCESSFUL
        print("wihtin-fov detection processing successful")
        # see belief now
        self.get_and_visualize_belief(o3dviz=o3dviz)
        self.get_and_visualize_robot_pose()


        rospy.spin()

if __name__ == "__main__":
    ProcessDetectionObservationTestCase(
        node_name="test_get_robot_belief_3d_with_point_cloud",
        debug=False).run()
