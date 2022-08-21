# This test builds upon the planner creation test (test_create_planner_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_plan_action_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import numpy as np
from sloop_object_search.grpc.utils import proto_utils as pbutil
from sloop_object_search.grpc.common_pb2 import Status
from test_create_planner_3d_with_point_cloud import CreatePlannerTestCase

class PlanActionTestcase(CreatePlannerTestCase):
    def run(self):
        super().run()
        response = self._sloop_client.planAction(
            self.robot_id,
            header=pbutil.make_header(frame_id=self.world_frame))
        assert response.status == Status.SUCCESSFUL
        action = pbutil.interpret_planned_action(response)
        print("plan action test passed. Action planned:")
        print(action)

if __name__ == "__main__":
    PlanActionTestcase(node_name="test_plan_action_3d_with_point_cloud",
                       debug=False).run()