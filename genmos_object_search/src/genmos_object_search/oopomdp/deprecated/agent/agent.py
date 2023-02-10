from .basic2d import SloopMosBasic2DAgent
from .topo2d import SloopMosTopo2DAgent

# deprecated
def make_agent(_config, init_pose=None, grid_map=None):
    print("Deprecation warning: 'make_agent' is no longer the supported"\
          "method for creating an agent")
    map_name = _config['task_config']["map_name"]

    if init_pose is None:
        raise ValueError("You must provide initial pose")
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"], map_name, grid_map=grid_map)
    return agent
