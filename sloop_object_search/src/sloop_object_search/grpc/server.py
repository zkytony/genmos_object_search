from concurrent import futures
import logging

import grpc
import sloop_object_search_pb2 as slpb2
import sloop_object_search_pb2_grpc

import yaml
from sloop_object_search.oopomdp.agent import make_agent as make_sloop_mos_agent


class SloopObjectSearchServer(sloop_object_search_pb2_grpc.SloopObjectSearchServer):
    def __init__(self):
        self._agents = {}

    def CreateAgent(self, request, context):
        if request.agent_name in self._agent:
            return slpb2.CreateAgentReply(status=slpb2.Status.FAILED,
                                          message=f"Agent with name {request.agent_name} already exists!")

        config_str = request.config.decode("utf-8")
        agent_config = yaml.safe_load(config_str)

        agent = make_sloop_mos_agent(agent_config)
