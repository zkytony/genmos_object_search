# Single object

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

config = {
    "planner_config": {
        "planner": "sloop_object_search.oopomdp.deprecated.planner.hier2d.HierarchicalPlanner",
        "high_level_planner_args": {
            "max_depth": 20,
            "exploration_const": 1000,
            "planning_time": 0.25
        },
        "local_search": {
            "planner": "pomdp_py.POUCT",
            "planner_args": {
                "max_depth": 10,
                "exploration_const": 1000,
                "planning_time": 0.15
            }
        }
    },

    "task_config": {
        "max_steps": 100,
        "map_name": "austin",
        "visualizer": "sloop_object_search.oopomdp.deprecated.agent.VizSloopMosTopo",
        "viz_params": {
            "res": 20
        },

    },

    "agent_config": {
        "agent_class": "SloopMosTopo2DAgent",  # This agent works in 2D grids
        "topo_map_args": {
            "num_place_samples": 10,
            "degree": [3, 5],
            "sep": 4.0,   # separation between places
            "seed": 1120,
            "node_coverage_radius": 3.0,
            # if the topo graph covers less than this amount of prob, resample
            "resample_prob_thres": 0.4
        },
        "topo_trans_args": {
            "h_angle_res": 45.0
        },
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
                    "class": "sloop_object_search.oopomdp.StaticObjectTransitionModel"
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
                    "class": "sloop_object_search.oopomdp.FanModelSimpleFP",
                    "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1, 0.25))
                },
            },
            "primitive_moves": {
                "func": "sloop_object_search.oopomdp.domain.action.basic_discrete_moves2d",
                "params": {
                    "step_size": 3,
                    "h_rotation": 45.0
                }
            }
        },
        "spacy_model": "en_core_web_lg",
        "foref_models_dir": os.path.join(ABS_PATH, "../../../models")
    },
}
