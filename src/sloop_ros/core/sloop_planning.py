def create_agent(belief, models, config):
    sloop_agent = eval(config["agent"])(
        belief,
        models.transition_model,
        models.observation_model,
        models.reward_model,
        models.policy_model
        **config["agent_params"]
    )
    return sloop_agent


def create_planner(config):
    planner = eval(config["planner"])(**config["planner_config"])
    return planner


def plan_next(planner, agent):
    return planner.plan(agent.belief)
