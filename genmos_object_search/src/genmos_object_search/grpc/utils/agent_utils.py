import json

import logging
import pomdp_py
import time
import copy

from genmos_object_search.oopomdp.agent import\
    (MosAgentBasic2D, MosAgentTopo2D, MosAgentBasic3D, MosAgentTopo3D)

import genmos_object_search.oopomdp.domain.observation as slpo
import genmos_object_search.oopomdp.domain.action as slpa
from genmos_object_search.oopomdp.models import belief
from genmos_object_search.oopomdp.models.search_region import SearchRegion2D, SearchRegion3D
from genmos_object_search.oopomdp.planner.hier import HierPlanner
from genmos_object_search.utils.misc import import_class

from . import proto_utils
from .search_region_processing import (search_region_2d_from_point_cloud,
                                       search_region_3d_from_point_cloud)
from ..constants import Message, Info


VALID_AGENTS = {"MosAgentBasic2D",
                "MosAgentTopo2D",
                "MosAgentBasic3D",
                "MosAgentTopo3D"}


def create_agent(robot_id, agent_config_world, robot_localization_world, search_region, **kwargs):
    """
    Creates a GenMOS POMDP agent named 'robot_id', with the given
    config (dict). The initial pose, in world frame, is given by
    robot_pose. The search_region can be 2D or 3D.

    Fields in agent_config: see example.

    Note that typically size units in agent_config that are metric.
    See _convert_metric_fields_to_pomdp_fields.
    """
    _validate_agent_config(agent_config_world)
    agent_config_pomdp = _convert_metric_fields_to_pomdp_fields(
        agent_config_world, search_region)
    agent_class = eval(agent_config_pomdp["agent_class"])

    # need to convert robot localization from world frame to pomdp frame
    robot_pose_world = robot_localization_world.pose
    robot_pose_cov_world = robot_localization_world.cov
    robot_pose_pomdp, robot_pose_cov_pomdp =\
        search_region.to_pomdp_pose(robot_pose_world, robot_pose_cov_world)
    # create initial robot pose dist
    init_robot_pose_dist = belief.RobotLocalization(robot_id, robot_pose_pomdp, robot_pose_cov_pomdp)
    if isinstance(search_region, SearchRegion2D):
        init_robot_pose_dist = init_robot_pose_dist.to_2d()
    agent = agent_class(agent_config_pomdp, search_region, init_robot_pose_dist, **kwargs)
    return agent


def _validate_agent_config(agent_config):
    if "agent_class" not in agent_config:
        raise KeyError("'agent_class' must exist agent_config")
    if "robot" not in agent_config:
        raise KeyError("'robot' must exist agent_config")
    if "objects" not in agent_config:
        raise KeyError("'objects' must exist agent_config")
    if "targets" not in agent_config:
        raise KeyError("'targets' must exist agent_config")
    agent_config['no_look'] = agent_config.get('no_look', True)
    if agent_config['agent_class'] not in VALID_AGENTS:
        raise ValueError(f"Agent class {agent_config['agent_class']} not recognized.")

    # Check if targets are valid objects
    for target_id in agent_config["targets"]:
        if target_id not in agent_config["objects"]:
            raise ValueError(f"target {target_id} is not a defined object.")

    # check if certain robot configs are present
    if "action" not in agent_config["robot"]:
        raise ValueError("Requires 'action' to be set in agent_config['robot'].")
    if "id" not in agent_config["robot"]:
        raise ValueError("Requires 'id' to be set in agent_config['robot'].")

    # Check if detectors are for valid objects
    if "detectors" not in agent_config["robot"]:
        raise ValueError(f"No detector specified.")
    for objid in agent_config["robot"]["detectors"]:
        if objid not in agent_config["objects"]:
            raise ValueError(f"detectable object {objid} is not a defined object.")


def _convert_metric_fields_to_pomdp_fields(agent_config_world, search_region):
    """POMDP agent takes in config in POMDP space. We do this conversion for detector specs."""
    def _convert_sensor_params(sensor_params_world, sensor_params_pomdp):
        if "max_range" in sensor_params_world:
            sensor_params_pomdp["max_range"] =\
                sensor_params_world["max_range"] / search_region.search_space_resolution
        if "min_range" in sensor_params_world:
            sensor_params_pomdp["min_range"] =\
                sensor_params_world["min_range"] / search_region.search_space_resolution
        if "near" in sensor_params_world:
            sensor_params_pomdp["near"] =\
                sensor_params_world["near"] / search_region.search_space_resolution
        if "far" in sensor_params_world:
            sensor_params_pomdp["far"] =\
                sensor_params_world["far"] / search_region.search_space_resolution

    agent_config_pomdp = copy.deepcopy(agent_config_world)
    for objid in agent_config_world["robot"]["detectors"]:
        sensor_params_world = agent_config_world["robot"]["detectors"][objid]["params"]["sensor"]
        sensor_params_pomdp = agent_config_pomdp["robot"]["detectors"][objid]["params"]["sensor"]
        _convert_sensor_params(sensor_params_world, sensor_params_pomdp)

    if len(agent_config_world["robot"].get("sensors", [])) > 0:
        for i in range(len(agent_config_world["robot"]["sensors"])):
            sensor_params_world = agent_config_world["robot"]["sensors"][i]["params"]
            sensor_params_pomdp = agent_config_pomdp["robot"]["sensors"][i]["params"]
            _convert_sensor_params(sensor_params_world, sensor_params_pomdp)

    # Convert prior
    if "prior" in agent_config_world["belief"]:
        for objid in agent_config_world["belief"]["prior"]:
            object_prior_world = agent_config_world["belief"]["prior"].get(objid, [])
            object_prior_pomdp = []
            if isinstance(search_region, SearchRegion3D):
                for voxel_world, prob in object_prior_world:
                    voxel_pos_pomdp = search_region.to_pomdp_pos(voxel_world[:3])
                    if len(voxel_world) == 3:
                        voxel_res_pomdp = 1
                    else:
                        voxel_res_pomdp = voxel_world[3] / search_region.search_space_resolution
                    voxel_pomdp = (*voxel_pos_pomdp, voxel_res_pomdp)
                    object_prior_pomdp.append([voxel_pomdp, prob])
            else:
                # 2D
                for pos, prob in object_prior_world:
                    pos_pomdp = search_region.to_pomdp_pos(pos[:2])
                    object_prior_pomdp.append([pos_pomdp, prob])

            agent_config_pomdp["belief"]["prior"][objid] = object_prior_pomdp

    # Convert step size in action
    if "action" in agent_config_world["robot"]:
        if "params" in agent_config_world["robot"]["action"]\
           and "step_size" in agent_config_world["robot"]["action"]["params"]:
            step_size_world = agent_config_world["robot"]["action"]["params"]["step_size"]
            step_size_pomdp = step_size_world / search_region.search_space_resolution
            agent_config_pomdp["robot"]["action"]["params"]["step_size"] = step_size_pomdp

    # Convert reachable parameters
    if "reachable" in agent_config_world["robot"]:
        if "min_height" in agent_config_world["robot"]["reachable"]:
            min_height_world = agent_config_world["robot"]["reachable"]["min_height"]
            min_height_pomdp = min_height_world / search_region.search_space_resolution
            agent_config_pomdp["robot"]["reachable"]["min_height"] = min_height_pomdp
        if "max_height" in agent_config_world["robot"]["reachable"]:
            max_height_world = agent_config_world["robot"]["reachable"]["max_height"]
            max_height_pomdp = max_height_world / search_region.search_space_resolution
            agent_config_pomdp["robot"]["reachable"]["max_height"] = max_height_pomdp

    # Convert topo config
    if "topo" in agent_config_world["robot"]["action"]:
        topo_config_world = agent_config_world["robot"]["action"]["topo"]
        topo_config_pomdp = agent_config_pomdp["robot"]["action"]["topo"]
        if "sep" in agent_config_world["robot"]["action"]["topo"]:
            sep_world = agent_config_world["robot"]["action"]["topo"]["sep"]
            sep_pomdp = sep_world / search_region.search_space_resolution
            agent_config_pomdp["robot"]["action"]["topo"]["sep"] = sep_pomdp

        if "sample_space" in topo_config_world:
            sample_space_world = topo_config_world["sample_space"]
            center_world = (sample_space_world["center_x"],
                            sample_space_world["center_y"],
                            sample_space_world["center_z"])
            center_pomdp = search_region.to_pomdp_pos(center_world)
            topo_config_pomdp["sample_space"]["center_x"] = center_pomdp[0]
            topo_config_pomdp["sample_space"]["center_y"] = center_pomdp[1]
            topo_config_pomdp["sample_space"]["center_z"] = center_pomdp[2]
            for field_name in {"size_x", "size_y", "size_z"}:
                if field_name in topo_config_world["sample_space"]:
                    field_world = topo_config_world["sample_space"][field_name]
                    field_pomdp = field_world / search_region.search_space_resolution
                    topo_config_pomdp["sample_space"][field_name] = field_pomdp

        if "3d_proj_2d" in topo_config_world:
            if "layout_cut" in topo_config_world["3d_proj_2d"]:
                layout_cut_world = topo_config_world["3d_proj_2d"]["layout_cut"]
                layout_cut_pomdp = layout_cut_world / search_region.search_space_resolution
                topo_config_pomdp["3d_proj_2d"]["layout_cut"] = layout_cut_pomdp
            if "floor_cut" in topo_config_world["3d_proj_2d"]:
                floor_cut_world = topo_config_world["3d_proj_2d"]["floor_cut"]
                floor_cut_pomdp = floor_cut_world / search_region.search_space_resolution
                topo_config_pomdp["3d_proj_2d"]["floor_cut"] = floor_cut_pomdp
            if "brush_size" in topo_config_world["3d_proj_2d"]:
                brush_size_world = topo_config_world["3d_proj_2d"]["brush_size"]
                brush_size_pomdp = brush_size_world / search_region.search_space_resolution
                topo_config_pomdp["3d_proj_2d"]["brush_size"] = brush_size_pomdp
            if "inflation" in topo_config_world["3d_proj_2d"]:
                inflation_world = topo_config_world["3d_proj_2d"]["inflation"]
                inflation_pomdp = inflation_world / search_region.search_space_resolution
                topo_config_pomdp["3d_proj_2d"]["inflation"] = inflation_pomdp

        # inflation for topo2d
        if "inflation" in topo_config_world:
            inflation_world = topo_config_world["inflation"]
            inflation_pomdp = inflation_world / search_region.search_space_resolution
            topo_config_pomdp["inflation"] = inflation_pomdp

    # Additional belief params
    if "init_params" in agent_config_world["belief"]:
        if "occupancy_height_thres" in agent_config_world["belief"]["init_params"]:
            oht_world = agent_config_world["belief"]["init_params"]["occupancy_height_thres"]
            oht_pomdp = oht_world / search_region.search_space_resolution
            agent_config_pomdp["belief"]["init_params"]["occupancy_height_thres"] = oht_pomdp

    return agent_config_pomdp

def voxel_to_world(v, search_region):
    """Given a voxel (x,y,z,r) where (x,y,z) is a point
    at the space with resolution r, return a voxel (x',y',z',r')in
    the world frame such that (x',y',z') refers to a point in
    the world frame, and r' is the size of the voxel in meters."""
    x,y,z,r = v
    # Convert the voxel to its ground level origin, then convert to world frame
    world_pos = search_region.to_world_pos((x*r, y*r, z*r))
    world_res = r * search_region.search_space_resolution
    return [*world_pos, world_res]

def update_belief(agent, observation, action=None, debug=False, **kwargs):
    """
    performs belief update and returns what the request wants.

    Args:
        request (ProcessObservationRequest)
        agent: pomdp agent
        observation: pomdp Observation
        action: pomdp action
        kwargs: contains parameters interpreted from request, such as 'return_fov'
    """
    _start_time = time.time()
    result = {}

    # Perform the belief update
    ret = agent.update_belief(
        observation, action=action, debug=debug, **kwargs)

    # Process auxiliary returning information
    if isinstance(observation, slpo.JointObservation):
        if isinstance(agent, MosAgentBasic3D):
            fovs = None
            if type(ret) == dict:
                fovs = ret.get("fovs")
            if fovs is not None:
                # serialize fovs as json string
                fovs_dict = {}
                convert_func = lambda v: voxel_to_world(v, agent.search_region)
                for objid in fovs:
                    visible_volume, obstacles_hit = fovs[objid]
                    fovs_dict[objid] = {"visible_volume": list(map(convert_func, visible_volume)),
                                        "obstacles_hit":  list(map(convert_func, obstacles_hit))}
                json_str = json.dumps(fovs_dict)
                result = {"fovs": json_str.encode(encoding='utf-8')}

    _total_time = time.time() - _start_time
    logging.info("Belief update took: {:.4f}s".format(_total_time))
    if kwargs.pop("return_raw_aux", False):
        return result, ret
    else:
        return result


def create_agent_search_region(agent_config, request):
    """Called before the agent is created, upon receiving an UpdateSearchRegion
    request. Returns a tuple, (SearchRegion, RobotLocalization), where the robot
    localization is in the world frame.
    """
    robot_loc_world = proto_utils.robot_localization_from_proto(request.robot_pose)
    robot_pose = robot_loc_world.pose  # world frame robot pose
    if agent_config["agent_class"] in {"MosAgentBasic2D", "MosAgentTopo2D"}:
        # 2D
        params = proto_utils.process_search_region_params_2d(
            request.search_region_params_2d)
        robot_position = robot_pose[:2]
        logging.info("converting point cloud to 2d search region...")
        search_region = search_region_2d_from_point_cloud(
            request.point_cloud, robot_position,
            **params)

    elif agent_config["agent_class"] in {"MosAgentBasic3D", "MosAgentTopo3D"}:
        # 3D
        params = proto_utils.process_search_region_params_3d(
            request.search_region_params_3d)
        robot_position = robot_pose[:3]
        logging.info("converting point cloud to 3d search region...")
        search_region = search_region_3d_from_point_cloud(
            request.point_cloud, robot_position,
            **params)
    else:
        raise ValueError(f"agent class unsupported: {agent_config['agent_class']}")
    # existing_search_region=self.search_region_for(request.robot_id),
    # existing_search_region=self.search_region_for(request.robot_id),
    return search_region, robot_loc_world


def update_agent_search_region(agent, request):
    robot_loc_world = proto_utils.robot_localization_from_proto(request.robot_pose)
    robot_pose = robot_loc_world.pose  # world frame robot pose

    if type(agent) in {MosAgentBasic2D, MosAgentTopo2D}:
        # 2D
        params = proto_utils.process_search_region_params_2d(
            request.search_region_params_2d)
        robot_position = robot_pose[:2]
        logging.info("converting point cloud to 2d search region...")
        search_region = search_region_2d_from_point_cloud(
            request.point_cloud, robot_position,
            existing_search_region=agent.search_region,
            **params)

    elif type(agent) in {MosAgentBasic3D, MosAgentTopo3D}:
        # 3D
        params = proto_utils.process_search_region_params_3d(
            request.search_region_params_3d)
        robot_position = robot_pose[:3]
        logging.info("converting point cloud to 3d search region...")
        search_region = search_region_3d_from_point_cloud(
            request.point_cloud, robot_position,
            existing_search_region=agent.search_region,
            **params)

    else:
        raise ValueError(f"agent class unsupported: {agent_config['agent_class']}")

    return search_region, robot_loc_world


def create_planner(planner_config, agent):
    """
    Creates a planner specified by 'planner_config' for
    the 'agent' (pomdp_py.Agent).
    """
    planner_class = planner_config["planner"]
    if planner_class.endswith("POUCT")\
       or planner_class.endswith("POMCP"):
        planner = import_class(planner_class)(**planner_config["planner_params"],
                                              rollout_policy=agent.policy_model)
        return planner
    elif planner_class.endswith("HierPlanner"):
        planner = import_class(planner_class)(agent, **planner_config["planner_params"])
        return planner
    else:
        raise NotImplementedError(f"Planner {planner_class} is unrecognized.")


def make_local_agent_config(hier_agent_config, belief_config=None):
    if belief_config is None:
        belief_config = hier_agent_config.get("belief_local", {})
    agent_config = {
        "agent_class": "MosAgentTopo3D",
        # 'local_hierarchical' means the agent is created as a local search agent
        # as part of hierarchical search.
        "agent_type": "local_hierarchical",
        "belief": belief_config,
        "robot": {
            "id": hier_agent_config["robot"]["id"] + "_local",
            "no_look": hier_agent_config["no_look"],
            "detectors": hier_agent_config["robot"]["detectors_local"],
            "sensors": hier_agent_config["robot"]["sensors_local"],
            "action": hier_agent_config["robot"]["action_local"],
            "reachable": hier_agent_config["robot"].get("reachable_local", {}),
            "color": hier_agent_config["robot"].get("color", [0.8, 0.2, 0.2]),
        },
        "objects": hier_agent_config["objects"],
        "targets": hier_agent_config["targets"]
    }
    return agent_config


def plan_action(planner, agent, server):
    """
    Performs planning and outputs the next action for execution.

    Note for hierarchical agent:
    When the planner of the global search agent outputs StayAction,
    the the following happens:
      (1) the server creates a placeholder for agent "hrobot0_local"
      (2) the server sends the client a message requesting a UpdateSearchRegion
          for "hrobot0_local". This is necessary in order to provide
          the search region to create the local search agent.
      (3) Upon receiving UpdateSearchRegion for "hrobot0_local", the
          local search agent is created.
      (4) The HierPlanner is given the local search agent, and plans
          an action for this agent to be executed by the client.

    Args:
        planner (pomdp_py.Planner)
        agent (MosAgent)
        server (grpc server)
    Returns:
        a tuple.
        The first element is a boolean. If planning is successful, return True
        The second element is either a str, reason for failure,
            or a tuple (bool, pomdp_py.Action); The first
            element is True if the agent planned locally.
    """
    action = planner.plan(agent)
    if hasattr(agent, "tree") and agent.tree is not None:
        # print planning tree
        print(pomdp_py.utils.typ.info(f"Search tree for {agent.robot_id}"))
        _dd = pomdp_py.utils.TreeDebugger(agent.tree)
        _dd.p(0)

    # If the agent is hierarchical, and it planned a Stay
    # action, then will ask the server to wait for an
    # update search region request in order to gather
    # the necessary inputs to create the local agent.
    if isinstance(planner, HierPlanner):
        if planner.global_agent.robot_id != agent.robot_id:
            return False, "Expecting planning request on behalf of global agent"

        if action.robot_id == agent.robot_id:
            if isinstance(action, slpa.StayAction):
                # The action is planned for the global agent, AND it is a Stay.
                # Handle the creation of local search agent if needed
                assert planner.local_agent is None,\
                    "Unexpected. Planner should output action for local agent"

                # server tells client, please send over update search region request
                local_robot_id = f"{agent.robot_id}_local"
                server.add_message(agent.robot_id,
                                   Message.REQUEST_LOCAL_SEARCH_REGION_UPDATE.format(local_robot_id))

                # Use the server's own mechanism to create the local agent
                # create a placeholder for the local agent in the server
                local_agent_config = make_local_agent_config(agent.agent_config)
                server.prepare_agent_for_creation(local_agent_config)

                # Now, wait for local search region; note that this will
                # also create the local agent, if it is not yet created.
                local_search_region, robot_loc_world =\
                    server.wait_for_client_provided_info(
                        Info.LOCAL_SEARCH_REGION.format(local_robot_id),
                        timeout=15)

                # Now, we have a local agent
                planner.set_local_agent(server.agents[local_robot_id])
                action = planner.plan_local()
            else:
                # The action is a subgoal that is not a Stay. We discard
                # the local agent, if there is one.
                if planner.local_agent is not None:
                    local_robot_id = f"{agent.robot_id}_local"
                    server.remove_agent(local_robot_id)
                    server.add_message(agent.robot_id,
                                       Message.LOCAL_AGENT_REMOVED.format(local_robot_id))

    planned_locally = False
    if isinstance(planner, HierPlanner):
        planned_locally = planner.planning_locally
    return True, (planned_locally, action)


def update_planner(planner, agent, observation, action):
    """update planner"""
    # For 3D agents, the planning observation for an object is a Voxel
    # at the ground resolution level. So we need to convert ObjectDetection
    # into a Voxel. For 2D agents, we need to convert it to ObjectLoc.
    if not isinstance(observation, slpo.GMOSObservation):
        raise TypeError(f"update_planner expects GMOSObservation, but got {type(observation)}.")
    planning_zobjs = {agent.robot_id: observation.z(agent.robot_id)}
    objects_found = agent.belief.b(agent.robot_id).mpe().objects_found
    for objid in observation:
        if objid == agent.robot_id:
            continue
        zobj = observation.z(objid)
        if isinstance(zobj, slpo.ObjectDetection):
            if agent.search_region.is_3d:
                if objid in objects_found:
                    planning_zobj = slpo.ObjectVoxel(objid, slpo.Voxel.NO_POSE, "found")
                if zobj.loc is not None:
                    planning_zobj = slpo.ObjectVoxel(objid, zobj.loc, objid)
                else:
                    planning_zobj = slpo.ObjectVoxel(objid, slpo.Voxel.NO_POSE,
                                                     slpo.Voxel.UNKNOWN)
            else:
                if objid in objects_found:
                    planning_zobj = slpo.ObjectLoc(objid, slpo.ObjectLoc.NO_LOC, "found")
                if zobj.loc is not None:
                    planning_zobj = slpo.ObjectLoc(objid, zobj.loc[:2], objid)
                else:
                    planning_zobj = slpo.ObjectLoc(objid,
                                                   slpo.ObjectLoc.NO_LOC,
                                                   slpo.ObjectLoc.UNKNOWN)
            planning_zobjs[objid] = planning_zobj
        else:
            raise TypeError(f"Unable to handle observation of type {zobj} for planner update")
    planning_observation = slpo.GMOSObservation(planning_zobjs)
    planner.update(agent, action, planning_observation)
    logging.info("planner updated")

def update_hier(request, planner, action, action_finished):
    """Update agent and planner (HierPlanner)."""
    if not isinstance(planner, HierPlanner):
        raise TypeError(f"update_hier only applies to HierPlanner. Got {type(planner)}")

    # If there is a local agent, will update its belief,
    # and then update the global agent's belief based on
    # the local agent belief.
    if planner.planning_locally:
        # Interpret request and update local agent belief.
        observation_local = proto_utils.pomdp_observation_from_request(
            request, planner.local_agent, action=action)
        aux_local, aux_raw = update_belief(planner.local_agent, observation_local, action=action,
                                           debug=request.debug, return_raw_aux=True,
                                           return_volumetric_observations=True,
                                           **proto_utils.process_observation_params(request))

        # Update global agent's object belief
        volumetric_observations = aux_raw["vobzs"]
        planner.update_global_object_beliefs_from_local(volumetric_observations)
        observation_global = proto_utils.pomdp_observation_from_request(
            request, planner.global_agent, action=action)
        robot_observation_global = observation_global.z(planner.global_agent.robot_id)
        aux_global = update_belief(planner.global_agent, robot_observation_global, action=action,
                                   debug=request.debug, **proto_utils.process_observation_params(request))
        aux = {**aux_local, **aux_global}

        assert isinstance(planner.last_planned_global_action, slpa.StayAction)
        update_planner(planner.local_planner, planner.local_agent, observation_local, action)
        update_planner(planner.global_planner, planner.global_agent, observation_global, planner.last_planned_global_action)

    else:
        # No local agent. Just update the global agent belief
        observation_global = proto_utils.pomdp_observation_from_request(
            request, planner.global_agent, action=action)
        aux = update_belief(planner.global_agent, observation_global, action=action,
                            **proto_utils.process_observation_params(request))
        update_planner(planner.global_planner, planner.global_agent, observation_global, planner.last_planned_global_action)

    return aux
