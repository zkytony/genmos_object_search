import pomdp_py
from sloop_ros.utils.misc import import_class


def create_agent(belief, models, config):
    agent_class = import_class(config["agent"])
    print(f"Creating Agent of class {agent_class}")
    sloop_agent = agent_class(
        belief, models, **config.get("agent_params", {}))
    return sloop_agent


def create_planner(config):
    planner = import_class(config["planner"])(**config.get("planner_config", {}))
    return planner


def plan_next(planner, agent):
    return planner.plan(agent.belief)
