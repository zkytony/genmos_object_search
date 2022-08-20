from concurrent import futures
import logging
import sys

import grpc

import argparse
import yaml
from . import sloop_object_search_pb2 as slpb2
from . import sloop_object_search_pb2_grpc as slbp2_grpc
from .common_pb2 import Status
from .utils import proto_utils as pbutil
from .utils import agent_utils
from .utils.search_region_processing import (search_region_2d_from_point_cloud,
                                             search_region_3d_from_point_cloud)


MAX_MESSAGE_LENGTH = 1024*1024*100  # 100MB


class SloopObjectSearchServer(slbp2_grpc.SloopObjectSearchServicer):
    def __init__(self):
        # maps from agent name to a pomdp_py.Agent.
        self._agents = {}

        # maps from agent name to a dictionary that contains
        # necessary arguments in order to construct the agent.
        self._pending_agents = {}

        self._world_origin = None

    def CreateAgent(self, request, context):
        """
        creates a SLOOP object search POMDP agent. Note that this agent
        is not yet be ready after this call. The server needs to wait
        for an "UpdateSearchRegionRequest".
        """
        if request.robot_id in self._agents:
            return slpb2.CreateAgentReply(
                header=pbutil.make_header(),
                status=Status.FAILED,
                message=f"Agent with name {request.robot_id} already exists!")

        config_str = request.config.decode("utf-8")
        config = yaml.safe_load(config_str)
        if "agent_config" in config:
            agent_config = config["agent_config"]
        else:
            agent_config = config
        self._pending_agents[request.robot_id] = {
            "agent_config": agent_config,
            "search_region": None,
            "init_robot_pose": None
        }

        return slpb2.CreateAgentReply(
            status=Status.PENDING,
            message="Agent configuration received. Waiting for additional inputs...")

    def GetAgentCreationStatus(self, request, context):
        if request.robot_id in self._pending_agents:
            return slpb2.GetAgentCreationStatusReply(
                header=pbutil.make_header(),
                status=Status.PENDING,
                status_message="Agent configuration received. Waiting for additional inputs...")
        elif request.robot_id not in self._agents:
            return slpb2.GetAgentCreationStatusReply(
                header=pbutil.make_header(),
                status=Status.FAILED,
                status_message="Agent does not exist.")
        elif request.robot_id in self._agents:
            return slpb2.GetAgentCreationStatusReply(
                header=pbutil.make_header(),
                status=Status.SUCCESS,
                status_message="Agent created.")
        else:
            raise RuntimeError("Internal error on GetAgentCreationStatus.")

    def UpdateSearchRegion(self, request, context):
        """If the agent in request is pending, then after this,
        the corresponding POMDP agent should be created and
        the agent is no longer pending. Otherwise, update the
        corresponding agent's search region."""

        robot_pose = pbutil.interpret_robot_pose(request)

        if request.HasField('occupancy_grid'):
            raise NotImplementedError()
        elif request.HasField('point_cloud'):
            params = {}
            if not request.is_3d:  # 2D
                params = pbutil.process_search_region_params_2d(
                    request.search_region_params_2d)
                robot_position = robot_pose[:2]
                logging.info("converting point cloud to 2d search region...")
                search_region = search_region_2d_from_point_cloud(
                    request.point_cloud, robot_position,
                    existing_search_region=self.search_region_for(request.robot_id),
                    **params)
            else: # 3D
                params = pbutil.process_search_region_params_3d(
                    request.search_region_params_3d)
                robot_position = robot_pose[:3]
                logging.info("converting point cloud to 3d search region...")
                search_region = search_region_3d_from_point_cloud(
                    request.point_cloud, robot_position,
                    existing_search_region=self.search_region_for(request.robot_id),
                    **params)
        else:
            raise ValueError("Either 'occupancy_grid' or 'point_cloud'"\
                             "must be specified in request.")

        # set search region for the agent for creation
        if request.robot_id in self._pending_agents:
            # prepare for creation
            self._pending_agents[request.robot_id]["search_region"] = search_region
            self._pending_agents[request.robot_id]["init_robot_pose"] = robot_pose
            self._create_agent(request.robot_id)

        elif request.robot_id in self._agents:
            # TODO: agent should be able to update its search region.
            raise NotImplementedError()

        else:
            logging.warn(f"Agent {request.robot_id} is not recognized.")

        return slpb2.UpdateSearchRegionReply(header=pbutil.make_header(),
                                             status=Status.SUCCESS,
                                             message="search region updated")

    def search_region_for(self, robot_id):
        if robot_id in self._pending_agents:
            return self._pending_agents[robot_id].get("search_region", None)
        elif robot_id in self._agents:
            return self._agents[robot_id].search_region
        else:
            return None

    def _create_agent(self, robot_id):
        """This function is called when an agent is first being created"""
        assert robot_id not in self._agents,\
            f"Internal error: agent {robot_id} already exists."
        info = self._pending_agents.pop(robot_id)
        self._agents[robot_id] = agent_utils.create_agent(
            robot_id, info["agent_config"], info["init_robot_pose"], info["search_region"])
        self._check_invariant()

    def _check_invariant(self):
        for robot_id in self._agents:
            assert robot_id not in self._pending_agents
        for robot_id in self._pending_agents:
            assert robot_id not in self._agents


###########################################################################
def serve(port=50051, max_message_length=MAX_MESSAGE_LENGTH):
    options = [('grpc.max_receive_message_length', max_message_length),
               ('grpc.max_send_message_length', max_message_length)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                                                    options=options)
    slbp2_grpc.add_SloopObjectSearchServicer_to_server(
        SloopObjectSearchServer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print("sloop_object_search started")
    server.wait_for_termination()

def main():
    parser = argparse.ArgumentParser(description="sloop object search gRPC server")
    parser.add_argument("--port", type=int, help="port, default 50051", default=50051)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    serve(args.port)

if __name__ == '__main__':
    main()
