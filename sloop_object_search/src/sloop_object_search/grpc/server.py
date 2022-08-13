from concurrent import futures
import logging

import grpc
import sloop_object_search.grpc.sloop_object_search_pb2 as slpb2
import sloop_object_search.grpc.sloop_object_search_pb2_grpc as slbp2_grpc
from sloop_object_search.grpc.common_pb2 import Status

import yaml
from sloop_object_search.oopomdp.agent import make_agent as make_sloop_mos_agent


MAX_MESSAGE_LENGTH = 1024*1024*100  # 100MB


class SloopObjectSearchServer(slbp2_grpc.SloopObjectSearchServicer):
    def __init__(self):
        self._agents = {}

    def CreateAgent(self, request, context):
        if request.agent_name in self._agents:
            return slpb2.CreateAgentReply(
                status=slpb2.Status.FAILED,
                message=f"Agent with name {request.agent_name} already exists!")

        config_str = request.config.decode("utf-8")
        agent_config = yaml.safe_load(config_str)
        print(agent_config)

        # agent = make_sloop_mos_agent(agent_config)
        return slpb2.CreateAgentReply(
            status=Status.SUCCESS,
            message=f"Creation of agent {request.agent_name} succeeded")


def serve(max_message_length=MAX_MESSAGE_LENGTH):
    options = [('grpc.max_receive_message_length', max_message_length),
               ('grpc.max_send_message_length', max_message_length)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                                                    options=options)
    slbp2_grpc.add_SloopObjectSearchServicer_to_server(
        SloopObjectSearchServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("sloop_object_search started")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
