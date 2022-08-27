from concurrent import futures
import logging
import sys

import grpc

import argparse
import yaml
import random
import pomdp_py

from . import sloop_object_search_pb2 as slpb2
from . import sloop_object_search_pb2_grpc as slbp2_grpc
from .common_pb2 import Status
from .utils import proto_utils
from .utils import agent_utils
from .utils import planner_utils
from .utils.search_region_processing import (search_region_2d_from_point_cloud,
                                             search_region_3d_from_point_cloud)
from ..utils.misc import hash16


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

        self._world_origin = None

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
                message=self._logwarn("Agent {request.robot_id} already exists!"))

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
            message=self._loginfo(f"Agent {request.robot_id} configuration received. Waiting for additional inputs..."))

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
        zrobotloc = proto_utils.robot_localization_from_proto(request.robot_pose)
        robot_pose = zrobotloc.pose  # world frame robot pose

        if request.HasField('occupancy_grid'):
            raise NotImplementedError()
        elif request.HasField('point_cloud'):
            params = {}
            if not request.is_3d:  # 2D
                params = proto_utils.process_search_region_params_2d(
                    request.search_region_params_2d)
                robot_position = robot_pose[:2]
                logging.info("converting point cloud to 2d search region...")
                search_region = search_region_2d_from_point_cloud(
                    request.point_cloud, robot_position,
                    existing_search_region=self.search_region_for(request.robot_id),
                    **params)
            else: # 3D
                params = proto_utils.process_search_region_params_3d(
                    request.search_region_params_3d)
                robot_position = robot_pose[:3]
                logging.info("converting point cloud to 3d search region...")
                search_region = search_region_3d_from_point_cloud(
                    request.point_cloud, robot_position,
                    existing_search_region=self.search_region_for(request.robot_id),
                    **params)
        else:
            raise RuntimeError("Either 'occupancy_grid' or 'point_cloud'"\
                               "must be specified in request.")

        # set search region for the agent for creation
        if request.robot_id in self._pending_agents:
            # prepare for creation
            self._pending_agents[request.robot_id]["search_region"] = search_region
            self._pending_agents[request.robot_id]["init_robot_localization"] = zrobotloc
            self._create_agent(request.robot_id)
            return slpb2.UpdateSearchRegionReply(header=proto_utils.make_header(),
                                                 status=Status.SUCCESSFUL,
                                                 message="Search region updated")

        elif request.robot_id in self._agents:
            # TODO: agent should be able to update its search region.
            return slpb2.UpdateSearchRegionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn("updating existing search region of an agent is not yet implemented"))

        else:
            logging.warning(f"Agent {request.robot_id} is not recognized.")
            return slpb2.UpdateSearchRegionReply(
                header=proto_utils.make_header(),
                status=Status.FAILED,
                message=self._logwarn(f"Agent {request.robot_id} is not recognized."))


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
            robot_id, info["agent_config"], info["init_robot_localization"], info["search_region"])
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
        planner = planner_utils.create_planner(planner_config, agent)
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
        action = planner.plan(agent)
        if hasattr(agent, "tree") and agent.tree is not None:
            # print planning tree
            _dd = pomdp_py.utils.TreeDebugger(agent.tree)
            _dd.p(0)
        header = proto_utils.make_header(request.header.frame_id)
        action_type, action_pb = proto_utils.pomdp_action_to_proto(action, agent, header)
        action_id = self._make_action_id(agent, action)
        self._actions_planned[agent.robot_id] = (action_id, action)
        return slpb2.PlanActionReply(header=header,
                                     action_id=action_id,
                                     **{action_type: action_pb})

    def _make_action_id(self, agent, action):
        action_id = "{}:{}_{}".format(agent.robot_id, action.name, self._action_seq)
        self._action_seq += 1
        return action_id

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
        robot_belief_pb = proto_utils.robot_belief_to_proto(robot_belief, agent.search_region, header)
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

        observation = proto_utils.pomdp_observation_from_request(request, agent, action=action)
        aux = agent_utils.update_belief(request, agent, observation, action=action)
        if action_finished:
            # update planner, when the action finishes
            planner = self._planners[request.robot_id]
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
