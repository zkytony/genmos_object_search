from .basic2d import SloopMosBasic2DAgent
from .topo2d import SloopMosTopo2DAgent

def make_agent(_config, init_pose=None):
    map_name = _config['task_config']["map_name"]

    if init_pose is None:
        init_pose = (5, 10, 0.0)
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"], map_name)
    return agent
