import pomdp_py
from sloop_object_search.utils.misc import import_class
from sloop_object_search.oopomdp.agent import HierMosAgent
from sloop_object_search.oopomdp.domain.action import StayAction


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


def plan_action(planner, agent, server):
    action = planner.plan(agent)
    if hasattr(agent, "tree") and agent.tree is not None:
        # print planning tree
        _dd = pomdp_py.utils.TreeDebugger(agent.tree)
        _dd.p(0)

    # If the agent is hierarchical, and it planned a Stay
    # action, then will ask the server to wait for an
    # update search region request in order to gather
    # the necessary inputs to create the local agent.
    if isinstance(agent, HierMosAgent)\
       and isinstance(action, StayAction):
        # server tells client, please send over update search region request
        server.add_message(agent.robot_id,
                           f"Request UpdateSearchRegion for {agent.robot_id}_local")

    return action
