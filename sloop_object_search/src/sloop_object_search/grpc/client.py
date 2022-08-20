# functions that make it easier to work with the server
#
# Convention:
#
#   - If the function involves calling (direct or indirect) gRPC
#     services, the function name is camelCase (first letter
#     lower-case).


import grpc
import yaml
import time

from . import sloop_object_search_pb2 as slpb2
from . import sloop_object_search_pb2_grpc as slpb2_grpc
from .common_pb2 import Pose2D, Status
from .server import MAX_MESSAGE_LENGTH
from .utils import proto_utils as pbutil


DEFAULT_RPC_TIMEOUT = 30


class SloopObjectSearchClient:

    def __init__(self, server_address='localhost:50051',
                 max_message_length=MAX_MESSAGE_LENGTH,
                 default_rpc_timeout=DEFAULT_RPC_TIMEOUT):
        self.server_address = server_address
        options = [('grpc.max_receive_message_length', max_message_length),
                   ('grpc.max_send_message_length', max_message_length)]
        # one client has one channel
        self.channel = grpc.insecure_channel(self.server_address, options=options)
        self.stub = slpb2_grpc.SloopObjectSearchStub(self.channel)

    def __del__(self):
        self.channel.close()

    def call(self, rpc_method, request, timeout=DEFAULT_RPC_TIMEOUT):
        return rpc_method(request, timeout=timeout)

    def call_async(self, rpc_method, request, timeout=DEFAULT_RPC_TIMEOUT):
        raise NotImplementedError()

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
            config_str = yaml.dump(config)
        else:
            with open(config_file_path) as f:
                config_str = f.read()

        timeout = kwargs.pop('timeout', DEFAULT_RPC_TIMEOUT)
        request = slpb2.CreateAgentRequest(
            config=config_str.encode(encoding='utf-8'),
            **kwargs)
        return self.call(self.stub.CreateAgent, request, timeout=timeout)

    def updateSearchRegion(self, **kwargs):
        timeout = kwargs.pop('timeout', DEFAULT_RPC_TIMEOUT)
        request = slpb2.UpdateSearchRegionRequest(**kwargs)
        return self.call(self.stub.UpdateSearchRegion, request, timeout=timeout)

    def getAgentCreationStatus(self, robot_id, **kwargs):
        timeout = kwargs.pop('timeout', DEFAULT_RPC_TIMEOUT)
        request = slpb2.GetAgentCreationStatusRequest(
            header=pbutil.make_header(),
            robot_id=robot_id)
        return self.call(self.stub.GetAgentCreationStatus, request, timeout=timeout)

    def waitForAgentCreation(self, robot_id, **kwargs):
        """blocks until an agent is created"""
        wait_sleep = kwargs.pop("wait_sleep", 0.2)
        response = self.getAgentCreationStatus(robot_id, **kwargs)
        while response.status != Status.SUCCESSFUL:
            response = self.getAgentCreationStatus(robot_id, **kwargs)
            time.sleep(wait_sleep)

    def createPlanner(self, config=None, **kwargs):
        config_str = yaml.dump(config)
        timeout = kwargs.pop('timeout', DEFAULT_RPC_TIMEOUT)
        request = slpb2.CreatePlannerRequest(
            config=config_str.encode(encoding='utf-8'),
            **kwargs
        )
        return self.call(self.stub.CreatePlanner, request, timeout=timeout)

    def planAction(self, robot_id, **kwargs):
        timeout = kwargs.pop('timeout', DEFAULT_RPC_TIMEOUT)
        request = slpb2.PlanActionRequest(
            robot_id=robot_id,
            **kwargs
        )
        return self.call(self.stub.PlanAction, request, timeout=timeout)
