
config = {

    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {}
    },

    "task_config": {
        "max_steps": 100,
        "map_name": "austin"
    },

    "agent_config": {
        "agent_class": "SloopMosBasic2DAgent",  # This agent works in 2D grids
        "action_scheme": "vw",
        "no_look": False,
        "belief": {
            "prior": "uniform"
        },
        "objects": {
            "targets": ["G"],
            "G": {
                "class": "GreenCar",
                "transition": {
                    "class": "StaticObjectTransitionModel"
                },
            },
            # "B": {
            #     "class": "RedBike",
            #     "transition": {
            #         "class": "StaticObjectTransitionModel"
            #     }
            # },
            # "R": {
            #     "class": "RedCar",
            #     "transition": {
            #         "class": "StaticObjectTransitionModel"
            #     },
            # },
        },
        "robot": {
            "id": "robot0",
            "detectors": {
                "G": {
                    "class": "FanModelSimpleFP",
                    "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1))
                },
                # "B": {
                #     "class": "FanModelSimpleFP",
                #     "params": (dict(fov=90, min_range=0, max_range=3), (0.8, 0.1))
                # },
                # "R": {
                #     "class": "FanModelSimpleFP",
                #     "params": (dict(fov=90, min_range=0, max_range=4), (0.9, 0.1))
                # },
            }
        },
        "spacy_model": "en_web_core_lg",
    },
}
