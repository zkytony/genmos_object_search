# Single object

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

config = {
    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "max_depth": 20,
            "exploration_const": 1000,
            "planning_time": 1.0
        }
    },

    "task_config": {
        "max_steps": 100,
        "map_name": "austin",
        "visualizer": "genmos_object_search.oopomdp.deprecated.agent.VizSloopMosBasic2D",
        "viz_params": {
            "res": 20
        },
    },

    "agent_config": {
        "agent_class": "SloopMosBasic2DAgent",  # This agent works in 2D grids
        "no_look": True,
        "belief": {
            # could be "groundtruth", "uniform", or "splang" (interactive)
            "prior": {"G": "splang"}
        },
        "targets": ["G"],
        "objects": {
            "G": {
                "class": "car",
                "transition": {
                    "class": "genmos_object_search.oopomdp.StaticObjectTransitionModel"
                },
                "color": [100, 200, 80]
            },
        },
        "object_symbol_map": {
            # Maps from object symbol to object id
            "GreenToyota": "G"
        },
        "robot": {
            "id": "robot0",
            "detectors": {
                "G": {
                    "class": "genmos_object_search.oopomdp.deprecated.FanModelSimpleFP",
                    "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1, 0.25))
                },
            },
            "primitive_moves": {
                "func": "genmos_object_search.oopomdp.domain.action.basic_discrete_moves2d",
                "params": {
                    "step_size": 3,
                    "h_rotation": 45.0
                }
            }
        },
        "spacy_model": "en_core_web_lg",
        "foref_models_dir": os.path.join(ABS_PATH, "../../../models"),
    },
}
