from sloop_object_search.oopomdp.agent import AGENT_CLASS_2D, AGENT_CLASS_3D,\
    SloopMosBasic2DAgent, MosBasic2DAgent, SloopMosTopo2DAgent, MosBasic3DAgent

VALID_AGENTS = {"SloopMosBasic2DAgent",
                "MosBasic2DAgent",
                "SloopMosTopo2DAgent",
                "MosBasic3DAgent"}

def create_agent(agent_name, agent_config_world, robot_pose, search_region):
    """
    Creates a SLOOP POMDP agent named 'agent_name', with the given
    config (dict). The initial pose, in world frame, is given by
    robot_pose. The search_region can be 2D or 3D.

    Fields in agent_config:

      General:
        "agent_class": (str)
        "robot": {"detectors": {<objid>: detectors_config},
                  "id": "robot_id",
                  "primitive_moves": primitive_moves_config,
                  "localization_model": (str; default 'identity')}
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
    _validate_agent_config(agent_config_pomdp)
    agent_config_pomdp = _convert_metric_fields_to_pomdp_fields(agent_config_world)
    agent_class = eval(agent_config_pomdp["agent_class"])




def _validate_agent_config(agent_config):
    if "agent_class" not in agent_config:
        raise KeyError("'agent_class' must exist agent_config")
    if "agent_class" not in VALID_AGENTS:
        raise ValueError(f"agent class {agent_config['agent_config']} is invalid")
    if "robot" not in agent_config:
        raise KeyError("'robot' must exist agent_config")
    if "objects" not in agent_config:
        raise KeyError("'objects' must exist agent_config")
    if "targets" not in agent_config:
        raise KeyError("'targets' must exist agent_config")
    agent_config['no_look'] = agent_config.get('no_look', True)
    if agent_config['agent_class'] not in AGENT_CLASS_2D\
       and agent_config['agent_class'] not in AGENT_CLASS_3D:
        raise ValueError(f"Agent class {agent_config['agent_class']} not recognized.")

    # Check if targets are valid objects
    for target_id in agent_config["targets"]:
        if target_id not in agent_config["objects"]:
            raise ValueError(f"target {target_id} is not a defined object.")

    # Check if detectors are for valid objects
    if "detectors" not in agent_config["robot"]:
        raise ValueError(f"No detector specified.")
    for objid in agent_config["robot"]["detectors"]:
        if objid not in agent_config["objects"]:
            raise ValueError(f"detectable object {objid} is not a defined object.")
        # If there is 'params', and it is a list of two elements, change it
        # to be a dict with 'sensor', 'quality'
        if "params" in agent_config["robot"]["detectors"][objid]:
            params = agent_config["robot"]["detectors"][objid]["params"]
            if type(params) == list:
                assert len(params) == 2,\
                    "'params' for detector model should be [sensor_params, quality_params]"
                params_dict = {"sensor": params[0], "quality": params[1]}
                agent_config["robot"]["detectors"][objid]["params"] = params_dict
            elif type(params) == dict:
                assert "sensor" in params,\
                    "'params' for detector model for {} doesn't have sensor params".format(objid)
                assert "quality" in params,\
                    "'params' for detector model for {} doesn't have quality params".format(objid)

def _convert_metric_fields_to_pomdp_fields(agent_config_world):
    """POMDP agent takes in config in POMDP space. We do this conversion for detector specs."""
    agent_config_pomdp = dict(agent_config_world)
    for objid in agent_config_world["robot"]["detectors"]:
        sensor_params_world = agent_config_world["robot"]["detectors"][objid]["sensor"]
        sensor_params_pomdp = agent_config_pomdp["robot"]["detectors"][objid]["sensor"]
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
    return agent_config_pomdp
