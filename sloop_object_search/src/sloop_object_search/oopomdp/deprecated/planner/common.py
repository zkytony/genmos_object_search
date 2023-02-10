from genmos_object_search.utils.misc import import_class

def make_planner(planner_config, agent):
    planner_class = planner_config['planner']
    if planner_class.startswith("pomdp_py"):
        return import_class(planner_class)(**planner_config["planner_params"],
                                           rollout_policy=agent.policy_model)
    elif planner_class.endswith("HierarchicalPlanner"):
        return import_class(planner_class)(planner_config, agent)
    else:
        raise ValueError(f"Cannot make planner {planner_class} (unrecognized).")
