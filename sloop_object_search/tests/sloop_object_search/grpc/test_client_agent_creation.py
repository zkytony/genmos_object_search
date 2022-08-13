import grpc

import sloop_object_search.grpc.sloop_object_search_pb2 as slpb2
import sloop_object_search.grpc.sloop_object_search_pb2_grpc as slpb2_grpc
from sloop_object_search.grpc.common_pb2 import Pose2D

def run():
    config_file = "./config_file_test_topo2d.yaml"
    with open(config_file) as f:
        config_str = f.read()

    create_agent_request = slpb2.CreateAgentRequest(
        agent_name="test_agent",
        config=config_str.encode(encoding='utf-8'),
        init_pose_2d=Pose2D(x=0, y=5, th=45.0),
    )

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = slpb2_grpc.SloopObjectSearchStub(channel)
        response = stub.CreateAgent(create_agent_request)
    print("Sloop Object Search client received: " + response.message)


if __name__ == "__main__":
    run()
