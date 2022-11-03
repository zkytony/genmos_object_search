# To run test (for UR5):
# 1. run in a terminal 'python -m sloop_object_search.grpc.server'
# 2. run in a terminal 'python sloop_mos_viam.py'
import yaml

from sloop_object_search.grpc.client import SloopObjectSearchClient
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.grpc import sloop_object_search_pb2 as slpb2
from sloop_object_search.grpc import observation_pb2 as o_pb2
from sloop_object_search.grpc import action_pb2 as a_pb2
from sloop_object_search.grpc import common_pb2
from sloop_object_search.grpc.common_pb2 import Status
from sloop_object_search.grpc.constants import Message
from sloop_object_search.utils.colors import lighter
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils.misc import import_class

def server_message_callback(message):
    print(message)

def run_sloop_search(config):
    """config: configuration dictionary for SLOOP"""
    sloop_client = SloopObjectSearchClient()
    agent_config = config["agent_config"]
    planner_config = config["planner_config"]
    robot_id = agent_config["robot"]["id"]

    last_action = None
    objects_found = set()

    # First, create an agent
    sloop_client.createAgent(
        header=proto_utils.make_header(), config=agent_config,
        robot_id=robot_id)

    # Make the client listen to server
    ls_future = sloop_client.listenToServer(
        robot_id, server_message_callback)
    local_robot_id = None  # needed if the planner is hierarchical

    # # Update search region
    # self.update_search_region()

    # # wait for agent creation
    # rospy.loginfo("waiting for sloop agent creation...")
    # self._sloop_client.waitForAgentCreation(self.robot_id)
    # rospy.loginfo("agent created!")

    # # visualize initial belief
    # self.get_and_visualize_belief()

    # # create planner
    # response = self._sloop_client.createPlanner(config=self.planner_config,
    #                                             header=proto_utils.make_header(),
    #                                             robot_id=self.robot_id)
    # rospy.loginfo("planner created!")

    # # Send planning requests
    # for step in range(self.config["task_config"]["max_steps"]):
    #     action_id, action_pb = self.plan_action()
    #     self.clear_fovs_markers()  # clear fovs markers before executing action
    #     self.execute_action(action_id, action_pb)
    #     ros_utils.WaitForMessages([self._action_done_topic], [std_msgs.String],
    #                               allow_headerless=True, verbose=True)
    #     rospy.loginfo(typ.success("action done."))

    #     if self.dynamic_update:
    #         self.update_search_region()

    #     response_observation, response_robot_belief =\
    #         self.wait_observation_and_update_belief(action_id)
    #     print(f"Step {step} robot belief:")
    #     robot_belief_pb = response_robot_belief.robot_belief
    #     objects_found = set(robot_belief_pb.objects_found.object_ids)
    #     self.objects_found.update(objects_found)
    #     print(f"  pose: {robot_belief_pb.pose.pose_3d}")
    #     print(f"  objects found: {objects_found}")
    #     print("-----------")

    #     # visualize FOV and belief
    #     self.get_and_visualize_belief()
    #     if response_observation.HasField("fovs"):
    #         self.visualize_fovs_3d(response_observation)

    #     # Check if we are done
    #     if objects_found == set(self.agent_config["targets"]):
    #         rospy.loginfo("Done!")
    #         break
    #     time.sleep(1)

def main():
    with open("./config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)
    run_sloop_search(config)

if __name__ == "__main__":
    main()
