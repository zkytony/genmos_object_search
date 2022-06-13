# Below is the config we'd like to have when running
# SLOOP object search in ROS

test_config = {
    "map_name": "austin",

    "agent_config": {
        "detectors": {
            "G": {
                "class": "FanModelSimpleFP",
                "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1))
            },
            "B": {
                "class": "FanModelSimpleFP",
                "params": (dict(fov=90, min_range=0, max_range=3), (0.8, 0.1))
            },
            "R": {
                "class": "FanModelSimpleFP",
                "params": (dict(fov=90, min_range=0, max_range=4), (0.9, 0.1))
            },
        },
        "no_look": False,
        "spacy_model": "en_web_core_lg",
        "agent_class": "SloopMosBasicAgent",  # This agent works in 2D grids
        "action_scheme": "vw"
    },

    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {}
    },

    "task_config": {
        "max_steps": 100
    }
}

def main(config):



    agent = ros_wrapped_agent(config)
    agent.run()


    mapinfo = MapInfoDataset()
    mapinfo.load_map(config["map_name"])
    grid_map = mapinfo.grid_map_of(config["map_name"])
    planner = eval(config["planner_config"])(**config["planner_params"],
                                             rollout_policy=agent.policy_model)

    agent = ROSifyAgent(eval(config["agent_class"])(config["agnet_config"]))
    agent.set_planner(planner)
    agent.setup()
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)
    max_steps = config["task_config"]["max_steps"]
    visualizer = ...
    for i in range(max_steps):
        action = send_plan_request
        reward = task_env.state_transition(action, execute=True)
        observation = task_env.provide_observation(agent.observation_model)
        send_observation(agent, obserfvation)
        visualizer...

if __name__ == "__main__":
    main(test_config)
