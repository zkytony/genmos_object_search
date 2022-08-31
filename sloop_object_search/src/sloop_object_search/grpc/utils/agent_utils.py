import json

import logging
import pomdp_py
import time
import copy

from sloop_object_search.oopomdp.agent import\
    (SloopMosAgentBasic2D, MosAgentBasic2D, SloopMosAgentTopo2D, MosAgentTopo2D,
     MosAgentBasic3D, MosAgentTopo3D)
import sloop_object_search.oopomdp.domain.observation as slpo
import sloop_object_search.oopomdp.domain.action as slpa
from sloop_object_search.oopomdp.agent import belief
from sloop_object_search.oopomdp.models.search_region import SearchRegion2D, SearchRegion3D

VALID_AGENTS = {"SloopMosAgentBasic2D",
                "MosAgentBasic2D",
                "SloopMosAgentTopo2D",
                "MosAgentTopo2D",
                "MosAgentBasic3D",
                "MosAgentTopo3D"}


def create_agent(robot_id, agent_config_world, robot_localization_world, search_region):
    """
    Creates a SLOOP POMDP agent named 'robot_id', with the given
    config (dict). The initial pose, in world frame, is given by
    robot_pose. The search_region can be 2D or 3D.

    Fields in agent_config:

      General:
        "agent_class": (str)
        "robot": {"detectors": {<objid>: detectors_config},
                  "id": "robot_id",
                  "action": action_config (e.g. primitive moves),
                  "localization_model": (str; default 'identity'),
                  "transition":  args for robot transition model}
        "objects": {<objid>: object_config}
        "targets": [<objid>]
        "no_look": (bool)

      Topo planning related:
        "topo_map_args" (optional):  refer to agents/topo2d.py
        "topo_trans_args" (optional):   refer to agents/topo2d.py

      Spatial language related:
        "spacy_model" (optional):  todo
        "foref_models_dir":  todo
        "foref_model_map_name":  todo
        "object_symbol_map": maps from object symbol (e.g. NovelBook) to object id (e.g. book)

    Note that by default, size units in agent_config that are metric.
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
    init_robot_pose_dist = belief.RobotPoseDist(robot_pose_pomdp, robot_pose_cov_pomdp)
    if isinstance(search_region, SearchRegion2D):
        init_robot_pose_dist = init_robot_pose_dist.to_2d()
    agent = agent_class(agent_config_pomdp, search_region, init_robot_pose_dist)
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
            agent_config_pomdp["belief"]["prior"][objid] = object_prior_pomdp

    # Convert step size in action
    if "action" in agent_config_world["robot"]:
        if "params" in agent_config_world["robot"]["action"]\
           and "step_size" in agent_config_world["robot"]["action"]["params"]:
            step_size_world = agent_config_world["robot"]["action"]["params"]["step_size"]
            step_size_pomdp = step_size_world / search_region.search_space_resolution
            agent_config_pomdp["robot"]["action"]["params"]["step_size"] = step_size_pomdp

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

def update_belief(request, agent, observation, action=None):
    """
    performs belief update and returns what the request wants.

    Args:
        request (ProcessObservationRequest)
        agent: pomdp agent
        observation: pomdp Observation
        action: pomdp action
    """
    _start_time = time.time()
    result = {}

    # Perform the belief update
    ret = agent.update_belief(observation, action=action, debug=request.debug,
                              return_fov=request.return_fov)

    # Process auxiliary returning information
    if isinstance(observation, slpo.JointObservation):
        if isinstance(agent, MosAgentBasic3D):
            fovs = ret
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
    return result


def update_planner(request, planner, agent, observation, action):
    """update planner"""
    # For 3D agents, the planning observation for an object is a Voxel
    # at the ground resolution level. So we need to convert ObjectDetection
    # into a Voxel.
    planning_zobjs = {agent.robot_id: observation.z(agent.robot_id)}
    for objid in observation:
        if objid == agent.robot_id:
            continue
        zobj = observation.z(objid)
        if isinstance(zobj, slpo.ObjectDetection):
            if zobj.loc is not None:
                planning_zobj = slpo.ObjectVoxel(objid, zobj.loc, objid)
            else:
                planning_zobj = slpo.ObjectVoxel(objid, slpo.Voxel.NO_POSE, slpo.Voxel.UNKNOWN)
            planning_zobjs[objid] = planning_zobj
        else:
            raise TypeError(f"Unable to handle observation of type {zobj} for planner update")
    planning_observation = slpo.GMOSObservation(planning_zobjs)
    planner.update(agent, action, planning_observation)
    logging.info("planner updated")
