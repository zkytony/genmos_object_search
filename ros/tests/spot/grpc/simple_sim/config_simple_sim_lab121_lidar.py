CONFIG = {
    "object_locations": {
        "book": [0.0, 2.0, 0.1],
        "cup": [1.2, 0.1, 0.5]
    },

    "agent_config": {
        "agent_class": "MosAgentTopo3D", #"MosAgentBasic3D",
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
                          # 'cup': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                          #          'params': {"sensor": "camera",
                          #                     "quality": [1e5, 0.1]}}},
            'action': {'func': 'sloop_object_search.oopomdp.domain.action.basic_discrete_moves3d',
                       'params': {'step_size': 0.2,
                                  'rotation': 90.0,
                                  'scheme': 'axis'}},
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]}},
                    # 'cup': {'class': 'cup',
                    #         'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                    #         'color': [0.89, 0.6, 0.05, 0.8],
                    #         'sizes': [0.12, 0.12, 0.12],
                    #         'viz_type': 'cube'}},
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

import yaml
def main():
    with open("config_simple_sim_lab121_lidar.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
