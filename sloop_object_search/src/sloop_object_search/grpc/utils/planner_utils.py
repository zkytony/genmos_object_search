import pomdp_py
from sloop_object_search.utils.misc import import_class

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
