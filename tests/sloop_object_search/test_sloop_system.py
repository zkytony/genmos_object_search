test_config = {
    "map_name": "austin",

    "agent_config": {
        "agent_class": "SloopMosBasic2DAgent",  # This agent works in 2D grids
        "action_scheme": "vw",
        "no_look": False,
        "belief": {
            "representation": "histogram",
            "prior": "uniform"
        },
        "objects": {
            "targets": ["G"],
            "G": {
                "transition": {
                    "class": "StaticObjectTransitionModel"
                },
            },
            "B": {
                "transition": {
                    "class": "StaticObjectTransitionModel"
                }
            },
            "R": {
                "transition": {
                    "class": "StaticObjectTransitionModel"
                },
            },
        },
        "robot": {
            "id": "robot0",
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
            }
        },
        "spacy_model": "en_web_core_lg",
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
    mapinfo = MapInfoDataset()
    mapinfo.load_map(config["map_name"])
    grid_map = mapinfo.grid_map_of(config["map_name"])
    agent = eval(config["agent_class"])(config["agnet_config"])

    init_state = ...
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)
    planner = eval(config["planner_config"])(**config["planner_params"],
                                             rollout_policy=agent.policy_model)

    max_steps = config["task_config"]["max_steps"]
    visualizer = ...
    for i in range(max_steps):
        action = planner.plan(agent)
        reward = task_env.state_transition(action, execute=True)
        observation = task_env.provide_observation(agent.observation_model)
        agent.update_belief(observation, action)
        visualizer...

if __name__ == "__main__":
    main(test_config)
