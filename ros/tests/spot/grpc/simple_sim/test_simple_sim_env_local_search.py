# Test 3D local search in SimpleSimEnv.
#
# To run the test:
# 1. 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=lab121_lidar'
# 2. 'roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=lab121_lidar'
#    This runs the SimpleSimEnv simulated environment.
# 3. 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose'
# 4. 'python -m sloop_object_search.grpc.server'
# 5. 'python test_simple_sim_env_local_search.py'
# ------------------
#
# We are testing the local search algorithm. We need to do:
# - Specify a scenario (object poses, robot starting pose).
# - Create an agent (rpc).
# - Visualize agent's belief
# - When the agent is ready, send planning request
# - Execute planned action
# - Synthesize observation
# - Visualize observation (e.g. FOV)
# - Update belief
# - Consider a large viewpoint-based action space.
# - Consider correlation
# - Consider realistic camera parameters
#
# Remember, you are a USER of the sloop_object_search package.
# Not its developer. You should only need to do basic things.
import rospy
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
from sloop_mos_ros import ros_utils
from sloop_object_search.grpc.client import SloopObjectSearchClient
from sloop_object_search.grpc.utils import proto_utils

REGION_POINT_CLOUD_TOPIC = "/spot_local_cloud_publisher/region_points"
INIT_ROBOT_POSE_TOPIC = "/simple_sim_env/init_robot_pose"

import yaml
with open("./config_simple_sim_lab121_lidar.yaml") as f:
    AGENT_CONFIG = yaml.safe_load(f)["agent_config"]


class TestSimpleEnvLocalSearch:
    def __init__(self):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node("test_simple_env_local_search")
        self._sloop_client = SloopObjectSearchClient()
        self.robot_id = AGENT_CONFIG["robot"]["id"]

        # First, create an agent
        self._sloop_client.createAgent(header=proto_utils.make_header(), config=AGENT_CONFIG,
                                       robot_id=self.robot_id)

        # need to get a region point cloud and a pose use that as search region
        region_cloud_msg, pose_stamped_msg = ros_utils.WaitForMessages(
            [REGION_POINT_CLOUD_TOPIC, INIT_ROBOT_POSE_TOPIC],
            [sensor_msgs.PointCloud2, geometry_msgs.PoseStamped],
            delay=10, verbose=True).messages
        cloud_pb = ros_utils.pointcloud2_to_pointcloudproto(region_cloud_msg)
        robot_pose_pb = proto_utils.posemsg_to_pose3dproto(pose_stamped_msg.pose)
        self._sloop_client.updateSearchRegion(header=cloud_pb.header,
                                              robot_id=self.robot_id,
                                              is_3d=True,
                                              robot_pose_3d=robot_pose_pb,
                                              point_cloud=cloud_pb,
                                              search_region_params_3d={"octree_size": 32,
                                                                       "search_space_resolution": 0.15,
                                                                       "debug": False})
        # wait for agent creation
        rospy.loginfo("waiting for sloop agent creation...")
        self._sloop_client.waitForAgentCreation(self.robot_id)
        rospy.loginfo("agent created!")

        # ...


def main():
    TestSimpleEnvLocalSearch()

if __name__ == "__main__":
    main()
