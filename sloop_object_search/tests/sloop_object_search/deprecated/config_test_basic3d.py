# Single object

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

config = {
    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "max_depth": 8,
            "exploration_const": 1000,
            "planning_time": 1.0
        }
    },

    "task_config": {
        "max_steps": 100,
    },

    "agent_config": {
        "agent_class": "MosBasic3DAgent",
        "no_look": True,
        "belief": {
            "prior": {"G": "uniform"}
        },
        "targets": ["G"],
        "objects": {
            "G": {
                "class": "car",
                "transition": {
                    "class": "sloop_object_search.oopomdp.StaticObjectTransitionModel"
                },
                "color": [100, 200, 80]
            }
        },
        "robot": {
            "id": "robot0",
            "detectors": {
                "G": {
                    "class": "sloop_object_search.oopomdp.FrustumVoxelAlphaBeta",
                    "params": (dict(fov=60, near=0.1, far=5), (1e5, 0.1)),
                }
            },
            "primitive_moves": {
                "func": "sloop_object_search.oopomdp.domain.action.basic_discrete_moves3d",
                "params": {
                    "step_size": 1,
                    "rotation": 90.0,
                    "scheme": "axis"
                }
            }
        }
    },
}
