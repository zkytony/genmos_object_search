# This test builds upon the planner creation test (test_create_planner_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m genmos_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_plan_action_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch genmos_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import numpy as np
from genmos_object_search.grpc.utils import proto_utils
from genmos_object_search.grpc.common_pb2 import Status
from genmos_object_search.grpc import observation_pb2 as o_pb2
from test_create_planner_3d_with_point_cloud import CreatePlannerTestCase

class PlanActionTestcase(CreatePlannerTestCase):
    def run(self):
        super().run()
        response = self._sloop_client.planAction(
            self.robot_id,
            header=proto_utils.make_header(frame_id=self.world_frame))
        assert response.status == Status.SUCCESSFUL
        action = proto_utils.interpret_planned_action(response)
        print("plan action success. Action planned:")
        print(action)
        print("action id:", response.action_id)
        action_id = response.action_id

        # Request again. This time should fail.
        response = self._sloop_client.planAction(
            self.robot_id,
            header=proto_utils.make_header(frame_id=self.world_frame))
        assert response.status == Status.FAILED

        # Now, mark the action to be finished - but with incorrect action id (should fail)
        robot_pose = (5, 1, 2, 0, 0, 0, 1)  # make up a pose
        robot_pose_pb = proto_utils.robot_pose_proto_from_tuple(robot_pose)
        header = proto_utils.make_header(frame_id=self.world_frame)
        detections_pb = o_pb2.ObjectDetectionArray(header=header, robot_id=self.robot_id, detections=[])
        response = self._sloop_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=detections_pb,
            header=header,
            action_id=action_id + "XXX",
            action_finished=True)
        assert response.status == Status.FAILED

        # Actual successful request to mark action finished
        header = proto_utils.make_header(frame_id=self.world_frame)
        detections_pb = o_pb2.ObjectDetectionArray(header=header, robot_id=self.robot_id, detections=[])
        response = self._sloop_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=detections_pb,
            header=header,
            action_id=action_id,
            action_finished=True)
        assert response.status == Status.SUCCESSFUL,\
            response.message
        print("action marked finished.")

        # If we do it again, we will fail
        header = proto_utils.make_header(frame_id=self.world_frame)
        detections_pb = o_pb2.ObjectDetectionArray(header=header, robot_id=self.robot_id, detections=[])
        response = self._sloop_client.processObservation(
            self.robot_id, robot_pose_pb,
            object_detections=detections_pb,
            header=header,
            action_id=action_id,
            action_finished=True)
        assert response.status == Status.FAILED
        print("test passed.")



if __name__ == "__main__":
    PlanActionTestcase(node_name="test_plan_action_3d_with_point_cloud",
                       debug=False).run()
