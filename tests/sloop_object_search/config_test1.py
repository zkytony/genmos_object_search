# Single object

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

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
        "targets": ["G"],
        "objects": {
            "G": {
                "class": "GreenCar",
                "transition": {
                    "class": "sloop_object_search.oopomdp.StaticObjectTransitionModel"
                },
            },
        },
        "robot": {
            "id": "robot0",
            "detectors": {
                "G": {
                    "class": "sloop_object_search.oopomdp.FanModelSimpleFP",
                    "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1, 0.25))
                },
            }
        },
        "spacy_model": "en_core_web_lg",
        "foref_models_dir": os.path.join(ABS_PATH, "../../models")
    },
}
