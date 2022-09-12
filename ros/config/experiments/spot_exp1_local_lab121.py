######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
CONFIG_LOCAL = {
    "object_locations": {
        "book": [0.0, 2.0, 0.1],
    },

    "agent_config": {
        "agent_class": "MosAgentTopo3D",
        "agent_type": "local",
        'belief': {"visible_volume_params": {"num_rays": 150,
                                             "step_size": 0.4,
                                             "voxel_res": 2},
                   "init_params": {"num_samples": 3000}},
        'robot': {
            'id': 'robot0',
            'no_look': True,
            'sensors': [{
                'name': 'camera',
                'params': {'fov': 61,
                           'far': 1.75,
                           'near': 0.2,
                           'occlusion_enabled': True}
            }],
            'detectors': {'book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e5, 0.1]}}},
            'action': {'topo': {'num_nodes': 10,
                                'pos_importance_thres': 0.01}},
            "reachable": {
                "min_height": 0.3,
                "max_height": 1.4
            }
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]}},
        'targets': ['book'],
    },

    "task_config": {
        "max_steps": 100
    },

    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "exploration_const": 1000,
            "max_depth": 8,
            "num_sims": 200,
            "show_progress": True
        }
    }
}

CONFIG = CONFIG_LOCAL

import yaml
def main():
    with open("spot_exp1_local_lab121.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
