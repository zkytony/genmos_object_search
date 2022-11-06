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
    import ObjectDetectionArray, Detection
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import euler_to_quat, quat_to_euler
from sloop_object_search.utils.open3d_utils import draw_octree_dist
from sloop_object_search.utils.colors import lighter
from sloop_object_search.oopomdp.models.octree_belief import plot_octree_belief

from test_create_planner_3d_with_point_cloud import CreatePlannerTestCase


class ProcessDetectionObservationTestCase(CreatePlannerTestCase):

    def _setup(self):
        super()._setup()
        self._robot_markers_pub = rospy.Publisher(
            "~robot_pose", MarkerArray, queue_size=10, latch=True)
        self._octbelief_markers_pub = rospy.Publisher(
            "~octree_belief", MarkerArray, queue_size=10, latch=True)
        self._object_markers_pub = rospy.Publisher(
            "~objects", MarkerArray, queue_size=10, latch=True)
        self._fovs_markers_pub = rospy.Publisher(
            "~fovs", MarkerArray, queue_size=10, latch=True)
        self.br = TransformBroadcaster()

    def visualize_fovs(self, response):
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        fovs = json.loads(response.fovs.decode('utf-8'))
        markers = []
        for objid in fovs:
            free_color = np.array(self.config["agent_config"]["objects"][objid].get(
                "color", [200, 100, 200]))/255
            hit_color = lighter(free_color*255, -0.25)/255

            obstacles_hit = set(map(tuple, fovs[objid]['obstacles_hit']))
            for voxel in fovs[objid]['visible_volume']:
                voxel = tuple(voxel)
                if voxel in obstacles_hit:
                    continue
                m = ros_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=free_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
            for voxel in obstacles_hit:
                m = ros_utils.make_viz_marker_for_voxel(
                    objid, voxel, header, color=hit_color, ns="fov",
                    lifetime=0, alpha=0.7)
                markers.append(m)
        self._fovs_markers_pub.publish(MarkerArray(markers))

    def get_and_visualize_belief(self, o3dviz=True):
        response = self._sloop_client.getObjectBeliefs(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        rospy.loginfo("got belief")

        # visualize the belief
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        markers = []
        for bobj_pb in response.object_beliefs:
            msg = ros_utils.make_octree_belief_proto_markers_msg(
                bobj_pb, header, alpha_scaling=1.0)
            self._octbelief_markers_pub.publish(msg)

        rospy.loginfo("belief visualized")

    def get_and_visualize_robot_pose(self):
        response = self._sloop_client.getRobotBelief(
            self.robot_id, header=proto_utils.make_header(self.world_frame))
        assert response.status == Status.SUCCESSFUL
        rospy.loginfo("got robot belief")
        marker, trobot = ros_utils.viz_msgs_for_robot_pose_proto(
            response.robot_belief.pose, self.world_frame, self.robot_id)
        self._robot_markers_pub.publish(MarkerArray([marker]))
        self.br.sendTransform(trobot)
        return response.robot_belief.pose

    def make_up_object(self, objid, object_loc, objsizes):
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        marker = ros_utils.make_viz_marker_for_object(
            objid, (*object_loc, 0, 0, 0, 1), header,
            scale=Vector3(*objsizes), lifetime=0)
        self._object_markers_pub.publish(MarkerArray([marker]))

        tobj = ros_utils.tf2msg_from_object_loc(
            object_loc, self.world_frame, objid)
        self.br.sendTransform(tobj)

    def test_one_round(self, objloc, objsizes=[0.75, 1.0, 0.75], o3dviz=True):

        detections = []
        if objloc is not None:
            target_id = self.config["agent_config"]["targets"][0]
            self.make_up_object(target_id, objloc, objsizes)
            objbox = Box3D(center=proto_utils.posetuple_to_poseproto((*objloc, 0, 0, 0, 1)),
                           sizes=Vec3(x=objsizes[0],
                                      y=objsizes[1],
                                      z=objsizes[2]))
            detections.append(Detection(label=target_id, box_3d=objbox))

        # Clear everything
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._object_markers_pub.publish(clear_msg)
        self._octbelief_markers_pub.publish(clear_msg)
        self._fovs_markers_pub.publish(clear_msg)
        self._robot_markers_pub.publish(clear_msg)

        # visualize belief now, with robot and object
        self.get_and_visualize_belief(o3dviz=o3dviz)
        robot_pose_pb = self.get_and_visualize_robot_pose()
        if objloc is not None:
            self.make_up_object(target_id, objloc, objsizes)

        robot_pose = proto_utils.robot_pose_from_proto(robot_pose_pb)

        # Send the process observation RPC request
        time.sleep(3)
        header_pb = proto_utils.make_header(self.world_frame)
        object_detection = ObjectDetectionArray(header=header_pb,
                                                robot_id=self.robot_id,
                                                detections=detections)
        response = self._sloop_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=object_detection,
            header=header_pb, return_fov=True)
        assert response.status == Status.SUCCESSFUL

        # see belief now, with robot and object, and fov
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        clear_msg = ros_utils.clear_markers(header, ns="")
        self._object_markers_pub.publish(clear_msg)
        self.visualize_fovs(response)
        self.get_and_visualize_belief(o3dviz=o3dviz)
        self.get_and_visualize_robot_pose()
        if objloc is not None:
            self.make_up_object(target_id, objloc, objsizes)


    def run(self, o3dviz=False):
        super().run()

        rospy.loginfo("testing no-detection processing")
        self.test_one_round(None, o3dviz=o3dviz)
        rospy.loginfo("no-detection processing done")

        rospy.loginfo("testing out-of-fov processing")
        objloc = [0.87, 1.92, 0.25]
        self.test_one_round(objloc, o3dviz=o3dviz)
        rospy.loginfo("out-of-fov detection processing successful")

        rospy.loginfo("testing within-fov processing")
        objloc = [4.20, 5.09, 0.45]
        self.test_one_round(objloc, o3dviz=o3dviz)
        rospy.loginfo("wihtin-fov detection processing successful")

        rospy.spin()

if __name__ == "__main__":
    ProcessDetectionObservationTestCase(
        node_name="test_get_robot_belief_3d_with_point_cloud",
        debug=False).run()
