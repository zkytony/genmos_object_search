from concurrent import futures
import logging
import sys

import grpc

import argparse
import yaml
import random
import pomdp_py
import time
from collections import deque

from . import sloop_object_search_pb2 as slpb2
from . import sloop_object_search_pb2_grpc as slbp2_grpc
from .common_pb2 import Status
from .utils import proto_utils
from .utils import agent_utils
from ..utils.misc import hash16
from .constants import Message, Info
from ..oopomdp.planner.hier import HierPlanner


MAX_MESSAGE_LENGTH = 1024*1024*100  # 100MB


class SloopObjectSearchServer(slbp2_grpc.SloopObjectSearchServicer):
    def __init__(self):
        # maps from robot id to a pomdp_py.Agent.
        self._agents = {}

        # maps from agent name to a dictionary that contains
        # necessary arguments in order to construct the agent.
        self._pending_agents = {}

        # planners. Maps from robot id to a planner (pomdp_py.Planner)
        self._planners = {}

        # maps from robot_id to (action_id, Action). Note that
        # once the robot finishes executing the action, its entry
        # will be removed.
        self._actions_planned = {}

        # maps from robot_id to {action_id -> Action}. Accumulated
        # as the robot finishes executing actions.
        self._actions_finished = {}

        # an sequence ID used for identifying an action
        self._action_seq = 0

        # messages to send to client
        self._messages_for_client = deque()

        # After sending a message to the client, the server may
        # expect to receive something from the client in return.
        # The temporary information sent from a client can be
        # saved here. The server is responsible for managing this.
        self._tmp_client_provided_info = {}


    def _loginfo(self, text):
        logging.info(text)
        return text

    def _logwarn(self, text):
        logging.warning(text)
        return text

    def CreateAgent(self, request, context):
        """
        creates a SLOOP object search POMDP agent. Note that this agent
        is not yet be ready after this call. The server needs to wait
        for an "UpdateSearchRegionRequest".
        """
        if request.robot_id in self._agents:
            return slpb2.CreateAgentReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"Agent {request.robot_id} already exists!"))

        config_str = request.config.decode("utf-8")
        config = yaml.safe_load(config_str)
        if "agent_config" in config:
            agent_config = config["agent_config"]
        else:
            agent_config = config
        if request.robot_id != agent_config["robot"]["id"]:
            return slpb2.CreateAgentReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"robot id in request ({request.robot_id}) "\
                                      "mismatch robot id in config {agent_config['robot']['id']}"))

        self.prepare_agent_for_creation(agent_config)
        return slpb2.CreateAgentReply(
            status=Status.PENDING,
            message=self._loginfo(f"Agent {request.robot_id} configuration received. Waiting for additional inputs..."))

    def prepare_agent_for_creation(self, agent_config,
                                   search_region=None,
                                   init_robot_loc=None):
        robot_id = agent_config["robot"]["id"]
        self._pending_agents[robot_id] = {
            "agent_config": agent_config,
            "search_region": search_region,
            "init_robot_loc": init_robot_loc
        }

    @property
    def agents(self):
        return self._agents

    def GetAgentCreationStatus(self, request, context):
        if request.robot_id in self._pending_agents:
            return slpb2.GetAgentCreationStatusReply(
                header=proto_utils.make_header(),
                status=Status.PENDING,
                status_message=f"Agent {request.robot_id} configuration received. Waiting for additional inputs...")
        elif request.robot_id not in self._agents:
            return slpb2.GetAgentCreationStatusReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                status_message=f"Agent {request.robot_id} does not exist.")
        elif request.robot_id in self._agents:
            return slpb2.GetAgentCreationStatusReply(
                header=proto_utils.make_header(),
                status=Status.SUCCESSFUL,
                status_message=f"Agent {request.robot_id} created.")
        else:
            raise RuntimeError("Internal error on GetAgentCreationStatus.")

    def UpdateSearchRegion(self, request, context):
        """If the agent in request is pending, then after this,
        the corresponding POMDP agent should be created and
        the agent is no longer pending. Otherwise, update the
        corresponding agent's search region."""
        if request.robot_id not in self._agents\
           and request.robot_id not in self._pending_agents:
            # We do not handle this robot_id
            return slpb2.UpdateSearchRegionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"Agent {request.robot_id} is not recognized."))

        if request.robot_id in self._agents:
            # This is an update search region request for an existing agent
            agent = self._agents[request.robot_id]
            search_region, robot_loc_world = \
                agent_utils.update_agent_search_region(agent, request)
            reply_msg = "Search region updated"

        elif request.robot_id in self._pending_agents:
            # This is an update search region request for a new agent
            agent_config = self._pending_agents[request.robot_id]["agent_config"]
            search_region, robot_loc_world =\
                agent_utils.create_agent_search_region(agent_config, request)
            self._pending_agents[request.robot_id]["search_region"] = search_region
            self._pending_agents[request.robot_id]["init_robot_loc"] = robot_loc_world
            self.create_agent(request.robot_id)
            reply_msg = f"Search region created (Agent {request.robot_id} created)"

        # provide info for local search, if needed
        _info_key = Info.LOCAL_SEARCH_REGION.format(request.robot_id)
        if self.waiting_for_client_provided_info(_info_key):
            self.add_client_provided_info(_info_key, (search_region, robot_loc_world))

        return slpb2.UpdateSearchRegionReply(header=proto_utils.make_header(),
                                             status=Status.SUCCESSFUL,
                                             message=self._loginfo(reply_msg))


    def search_region_for(self, robot_id):
        if robot_id in self._pending_agents:
            return self._pending_agents[robot_id].get("search_region", None)
        elif robot_id in self._agents:
            return self._agents[robot_id].search_region
        else:
            return None

    def create_agent(self, robot_id):
        """This function is called when an agent is first being created"""
        assert robot_id not in self._agents,\
            f"Internal error: agent {robot_id} already exists."
        info = self._pending_agents.pop(robot_id)
        self._agents[robot_id] = agent_utils.create_agent(
            robot_id, info["agent_config"], info["init_robot_loc"], info["search_region"])
        # if the agent is a local agent for hierarchical search, ensure its
        # search region is within the bound of the global agent's search region.
        if self._agents[robot_id].is_local_hierarchical:
            global_robot_id = robot_id.split("_local")[0]
            if global_robot_id not in self._agents:
                raise RuntimeError("Expecting global agent to be present, when creating local agent")
            global_agent = self._agents[global_robot_id]
            self._agents[robot_id].search_region.fit(global_agent.search_region)
        self._check_invariant()

    def _check_invariant(self):
        for robot_id in self._agents:
            assert robot_id not in self._pending_agents
        for robot_id in self._pending_agents:
            assert robot_id not in self._agents

    def CreatePlanner(self, request, context):
        """initializes the planner. The planner's parameters
        are contained in a encoded JSON/yaml string in the request.
        """
        if request.robot_id not in self._agents:
            # agent not yet created
            return slpb2.CreatePlannerReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"agent {request.robot_id} does not exist. Did you create it?"))

        if request.robot_id in self._planners:
            # a planner is already created. Change only if 'overwrite'
            if not request.overwrite:
                return slpb2.CreatePlannerReply(
                    header=proto_utils.make_header(),
                    status=Status.FAILED,
                    message=self._logwarn(f"Planner already exists for {request.robot_id}. Not overwriting."))

        config_str = request.config.decode("utf_8")
        config = yaml.safe_load(config_str)
        if "planner_config" in config:
            planner_config = config["planner_config"]
        else:
            planner_config = config
        agent = self._agents[request.robot_id]
        planner = agent_utils.create_planner(planner_config, agent)
        self._planners[request.robot_id] = planner

        return slpb2.CreatePlannerReply(
            header=proto_utils.make_header(),
            status=Status.SUCCESSFUL,
            message=self._loginfo(f"Planner created for {request.robot_id}"))

    def PlanAction(self, request, context):
        """The agent can only plan and execute one action at a time.
        Before a previously planned action is marked finished, no
        more planning will happen."""
        if request.robot_id not in self._agents:
            # agent not yet created
            return slpb2.PlanActionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"agent {request.robot_id} does not exist. Did you create it?"))

        if request.robot_id not in self._planners:
            # a planner is already created. Change only if 'overwrite'
            if not request.overwrite:
                return slpb2.PlanActionReply(
                    header=proto_utils.make_header(),
                    status=Status.FAILED,
                    message=self._logwarn(f"Planner does not exists for {request.robot_id}. Did you create it?"))

        if self._actions_planned.get(request.robot_id) is not None:
            action_id = self._actions_planned[request.robot_id][0]
            return slpb2.PlanActionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"Previously planned action {action_id} is not yet finished for {request.robot_id}."))

        agent = self._agents[request.robot_id]
        planner = self._planners[request.robot_id]
        success, result = agent_utils.plan_action(planner, agent, self)
        if not success:
            return slpb2.PlanActionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"Planning failed. {result}"))
        action = result
        header = proto_utils.make_header(request.header.frame_id)
        action_type, action_pb = proto_utils.pomdp_action_to_proto(action, agent, header)
        action_id = self._make_action_id(agent, action)
        self._actions_planned[agent.robot_id] = (action_id, action)
        return slpb2.PlanActionReply(header=header,
                                     status=Status.SUCCESSFUL,
                                     action_id=action_id,
                                     **{action_type: action_pb})

    def _make_action_id(self, agent, action):
        action_id = "{}:{}_{}".format(agent.robot_id, action.name, self._action_seq)
        self._action_seq += 1
        return action_id

    def wait_for_client_provided_info(self, key, timeout=None, sleep_time=1.0):
        """waits until the server receives information with key"""
        _start = time.time()
        self._loginfo(f"waiting for client to provide info for {key}")
        self._tmp_client_provided_info[key] = None
        while True:
            if self._tmp_client_provided_info[key] is not None:
                return self._tmp_client_provided_info[key]
            time.sleep(sleep_time)
            if timeout is not None:
                if time.time() - _start > timeout:
                    raise RuntimeError("Timeout when waiting for search region request")

    def add_client_provided_info(self, key, info):
        self._tmp_client_provided_info[key] = info

    def waiting_for_client_provided_info(self, key):
        """returns True if the server is currently waiting for a piece of info called 'key'
        from the client."""
        return key in self._tmp_client_provided_info\
            and self._tmp_client_provided_info[key] is None

    def GetObjectBeliefs(self, request, context):
        if request.robot_id not in self._agents:
            # agent not yet created
            return slpb2.GetObjectBeliefsReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"agent {request.robot_id} does not exist. Did you create it?"))

        agent = self._agents[request.robot_id]
        object_beliefs = {}
        if len(request.object_ids) > 0:
            object_beliefs = {objid: agent.belief.b(objid)
                              for objid in request.object_ids}
            if request.robot_id in object_beliefs:
                logging.warn("removing robot_id in object_ids in GetObjectBeliefs request")
                object_beliefs.pop(request.robot_id)  # remove belief about robot
        else:
            object_beliefs = dict(agent.belief.object_beliefs)
            object_beliefs.pop(request.robot_id)  # remove belief about robot

        object_beliefs_pb = proto_utils.pomdp_object_beliefs_to_proto(object_beliefs, agent.search_region)
        header = proto_utils.make_header(request.header.frame_id)
        return slpb2.GetObjectBeliefsReply(header=header,
                                           status=Status.SUCCESSFUL,
                                           message="got object beliefs",
                                           object_beliefs=object_beliefs_pb)

    def GetRobotBelief(self, request, context):
        if request.robot_id not in self._agents:
            # agent not yet created
            return slpb2.GetRobotBeliefReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"agent {request.robot_id} does not exist. Did you create it?"))

        agent = self._agents[request.robot_id]
        robot_belief = agent.belief.b(request.robot_id)
        header = proto_utils.make_header(request.header.frame_id)
        _other_fields = {}
        if hasattr(agent, "topo_map"):
            topo_map_pb = proto_utils.topo_map_to_proto(
                agent.topo_map, agent.search_region)
            _other_fields["topo_map"] = topo_map_pb
        robot_belief_pb = proto_utils.robot_belief_to_proto(
            robot_belief, agent.search_region, header,
            **_other_fields)
        return slpb2.GetRobotBeliefReply(
            header=header,
            status=Status.SUCCESSFUL,
            message=self._loginfo("got robot belief"),
            robot_belief=robot_belief_pb)

    def ProcessObservation(self, request, context):
        if request.robot_id not in self._agents:
            # agent not yet created
            return slpb2.ProcessObservationReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"agent {request.robot_id} does not exist. Did you create it?"))

        agent = self._agents[request.robot_id]
        action = None
        action_finished = False
        if request.HasField("action_id"):
            # make sure the action id here matches the action id of the planned action
            planned_action_id = self._planned_action_id(request.robot_id)
            if request.action_id != planned_action_id:
                # mismatch - this should not happen
                return slpb2.ProcessObservationReply(
                    header=proto_utils.make_header(),
                    status=Status.FAILED,
                    message=self._logwarn(f"action id mismatch. Action in request {request.action_id}"\
                                          f" is not the planned action {planned_action_id}"))
            action = self._actions_planned[request.robot_id][1]

            if request.HasField("action_finished") and request.action_finished:
                # mark action as finished
                if request.robot_id not in self._actions_finished:
                    self._actions_finished[request.robot_id] = {}
                self._actions_finished[request.robot_id][request.action_id] = action
                self._actions_planned.pop(request.robot_id)
                action_finished = True

        planner = self._planners[request.robot_id]
        if isinstance(planner, HierPlanner):
            # If the planner is hierarchical, we will let the planner
            # handle the belief update --> because it is more tricky
            # assert planner.global_agent.robot_id == agent.robot_id,\
            #     "Expecting request to contain global agent's robot id"
            # aux = agent_utils.update_hier(
            #     request, planner, action, action_finished)
            raise NotImplementedError()
        else:
            # Otherwise, we will update the agent and planner normally
            observation = proto_utils.pomdp_observation_from_request(request, agent, action=action)
            aux = agent_utils.update_belief(
                agent, observation, action=action,
                debug=request.debug, **proto_utils.process_observation_params(request))
            if action_finished:
                # update planner, when the action finishes
                agent_utils.update_planner(request, planner, agent, observation, action)

        header = proto_utils.make_header(request.header.frame_id)
        return slpb2.ProcessObservationReply(
            header=header,
            status=Status.SUCCESSFUL,
            message=self._loginfo(f"observation processed. Belief updated."),
            **aux)

    def _planned_action_id(self, robot_id):
        if robot_id not in self._actions_planned:
            return None
        else:
            return self._actions_planned[robot_id][0]

    def ListenServer(self, request, context):
        """This rpc allows client to receive messages from the server"""
        logging.info("server-client communication established.")
        while True:
            if len(self._messages_for_client) > 0:
                robot_id, message = self._messages_for_client[0]
                if robot_id != request.robot_id:
                    # the message is not intended for this request
                    continue
                else:
                    self._messages_for_client.popleft()
                header = proto_utils.make_header(request.header.frame_id)
                response = slpb2.ListenServerReply(header=header,
                                                   robot_id=robot_id,
                                                   message=message)
                logging.info(f"Sending {message} to {robot_id}")
                yield response

            else:
                # otherwise, sleep
                time.sleep(3)

    def add_message(self, robot_id, message):
        """adds a message in the queue to be sent to a client"""
        self._messages_for_client.append((robot_id, message))


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
