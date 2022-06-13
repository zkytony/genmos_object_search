from .framework.base_agent_wrapper import BaseAgentWrapper
from sloop_object_search.oopomdp.agent import make_planner
from sloop_object_search.oopomdp.planner import make_planner

def make_ros_agent(config):
    """
    Args:
        config (dict): the configuration dictionary; refer to examples
            under sloop_ros/tests
    """
    map_name = _config['task_config']["map_name"]

    init_robot_pose = (5, 10, 0.0)
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_robot_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"], map_name)

    _planner_config = _config["planner_config"]
    planner = make_planner(_planner_config, agent)
    return SloopMosROSAgentWrapper(agent, planner,
                                   config=config.get("ros_config", {}))


class SloopMosROSAgentWrapper(BaseAgentWrapper):
    def __init__(self, oopomdp_agent, planner, ros_config={}):
        self.planner = planner
        super().__init__(oopomdp_agent,
                         config=ros_config,
                         planner=planner)
