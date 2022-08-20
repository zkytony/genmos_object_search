# This test builds upon the agent creation test (test_create_agent_3d_with_point_cloud.py)
# To run the test
#
# 1. In one terminal, run 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>'
# 2. In another terminal, run 'python -m sloop_object_search.grpc.server'
# 3. In another terminal, run this test 'python test_create_planner_3d_with_point_cloud.py'
# 4. In another terminal, run 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/fake_robot_pose'
#
# Requires both point cloud and waypoints
import numpy as np
from sloop_object_search.grpc.utils import proto_utils as pbutil
from sloop_object_search.grpc.common_pb2 import Status
from test_create_agent_3d_with_point_cloud import CreateAgentTestCase

class CreatePlannerTestcase(CreateAgentTestCase):
    def run(self):
        super().run()

        planner_config = {
            "planner": "pomdp_py.POUCT",
            "planner_params": {
                "exploration_const": 1000,
                "max_depth": 10,
                "num_sims": 100
            }
        }
        response = self._sloop_client.createPlanner(config=planner_config,
                                                    header=pbutil.make_header(),
                                                    robot_id=self.robot_id)
        assert response.status == Status.SUCCESSFUL
        print("create planner test passed")



if __name__ == "__main__":
    CreatePlannerTestcase(node_name="test_create_planner_3d_with_point_cloud",
                        debug=False).run()
