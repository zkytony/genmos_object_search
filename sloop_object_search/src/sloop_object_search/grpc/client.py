# functions that make it easier to work with the server

import grpc
import json

import sloop_object_search.grpc.sloop_object_search_pb2 as slpb2
import sloop_object_search.grpc.sloop_object_search_pb2_grpc as slpb2_grpc
from sloop_object_search.grpc.common_pb2 import Pose2D


class SloopObjectSearchClient:

    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = slpb2_grpc.SloopObjectSearchStub(self.channel)

    def __del__(self):
        self.channel.close()

    def CreateAgent(self, config=None, config_file_path=None, **kwargs):
        """
        Calls the CreateAgent rpc. Sends over agent configuration,
        along with other parameters in kwargs. Agent configuration is
        a dictionary, which will be sent over as bytes of its string form.
        """
        if config_file_path is None:
            if config is None:
                raise ValueError("Agent config not specified."\
                                 "Either specify a file path or a dictionary.")
            if not type(config) == dict:
                raise ValueError("'config' should be a dict.")
            config_str = json.dumps(config)
        else:
            with open(config_file_path) as f:
                config_str = f.read()

        create_agent_request = slpb2.CreateAgentRequest(
            config=config_str.encode(encoding='utf-8'),
            **kwargs)
        response = self.stub.CreateAgent(create_agent_request)
        return response
