from concurrent import futures
import logging

import grpc
import sloop_object_search.grpc.sloop_object_search_pb2 as slpb2
import sloop_object_search.grpc.sloop_object_search_pb2_grpc as slbp2_grpc
from sloop_object_search.grpc.common_pb2 import Status

import yaml
from sloop_object_search.oopomdp.agent import make_agent as make_sloop_mos_agent
from sloop_object_search.oopomdp.agent import AGENT_CLASS_2D, AGENT_CLASS_3D



MAX_MESSAGE_LENGTH = 1024*1024*100  # 100MB


class SloopObjectSearchServer(slbp2_grpc.SloopObjectSearchServicer):
    def __init__(self):
        self._agents = {}
        self._world_origin = None

    def CreateAgent(self, request, context):
        """
        creates a SLOOP object search POMDP agent
        """
        if request.agent_name in self._agents:
            return slpb2.CreateAgentReply(
                status=slpb2.Status.FAILED,
                message=f"Agent with name {request.agent_name} already exists!")

        agent, response = create_agent.process_request(request)

        # agent = make_sloop_mos_agent(agent_config)
        return slpb2.CreateAgentReply(
            status=Status.SUCCESS,
            message=f"Creation of agent {request.agent_name} succeeded")

    def _create_agent(self, request):
        """
        Notes:
         - regarding 'init_pose':
            If the agent is 3D, the initial pose must be 3D. If the
            agent is 2D, the initial pose can be 2D or 3D

        Args:
            request (CreateAgentRequest)
        Returns:
            Agent, response
        """
        config_str = request.config.decode("utf-8")
        agent_config = yaml.safe_load(config_str)

        # We will first process the map
        if request.occupancy_grid is not None:
            search_region = search_region_from_occupancy_grid(
                request.occupancy_grid, agent_config)
        elif request.point_cloud is not None:
            search_region = search_region_from_point_cloud(
                request.occupancy_grid, agent_config)
        else:
            raise ValueError("Either 'occupancy_grid' or 'point_cloud'"\
                             "must be specified in request.")

        # Verify if initial pose is valid; If the agent is 3D,
        # the initial pose must be 3D. If the agent is 2D, the
        # initial pose can be 2D or 3D
        agent_class = agent_config["agent_class"]
        if agent_class in AGENT_CLASS_2D:
            if request.init_pose_2d is not None:
                init_pose = request.init_pose_2d
            elif request.init_pose_3d is not None:
                init_pose = request.init_pose_3d


        if agent_class in AGENT_CLASS_3D:
           assert request.init_pose_3d is not None


        import pdb; pdb.set_trace()





        print(agent_config)




###########################################################################
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
