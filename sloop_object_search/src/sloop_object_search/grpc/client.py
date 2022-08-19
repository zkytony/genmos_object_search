# functions that make it easier to work with the server
#
# Convention:
#
#   - If the function involves calling (direct or indirect) gRPC
#     services, the function name is camelCase (first letter
#     lower-case).


import grpc
import json

import sloop_object_search.grpc.sloop_object_search_pb2 as slpb2
import sloop_object_search.grpc.sloop_object_search_pb2_grpc as slpb2_grpc
from sloop_object_search.grpc.common_pb2 import Pose2D

from .server import MAX_MESSAGE_LENGTH
from .utils import proto_utils as pbutil


class SloopObjectSearchClient:

    def __init__(self, server_address='localhost:50051',
                 max_message_length=MAX_MESSAGE_LENGTH):
        self.server_address = server_address
        options = [('grpc.max_receive_message_length', max_message_length),
                   ('grpc.max_send_message_length', max_message_length)]
        self.channel = grpc.insecure_channel(self.server_address, options=options)
        self.stub = slpb2_grpc.SloopObjectSearchStub(self.channel)

    def __del__(self):
        self.channel.close()

    def createAgent(self, config=None, config_file_path=None, **kwargs):
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

    def updateSearchRegion(self, **kwargs):
        request = slpb2.UpdateSearchRegionRequest(**kwargs)
        response = self.stub.UpdateSearchRegion(request)
        return response

    def checkIfAgentIsCreated(self, agent_name):
        slpb2.GetAgentCreationStatusRequest()
