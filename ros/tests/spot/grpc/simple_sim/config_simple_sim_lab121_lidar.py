
######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
CONFIG_LOCAL = {
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



######### THE FOLLOWING IS USED FOR 2D LOCAL SEARCH TEST #########
CONFIG_LOCAL2D = {
    "object_locations": {
        "book": [0.0, 2.0, 0.1],
        "cup": [1.2, 0.1, 0.5]
    },

    "agent_config": {
        "agent_class": "MosAgentBasic2D",
        "agent_type": "local",  # 'hierarchical' or 'local'
        "belief": {},
        "robot": {
            "id": "robot0",
            "no_look": True,
            "sensors": [{"name": 'fan',
                         'params': {'fov': 61,
                                    'min_range': 0.2,
                                    'max_range': 1.75}}],
            'detectors': {'book': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                                   'params': {"sensor": "fan",
                                              "quality": [1e5, 0.05]}}},
            'color': [0.9, 0.1, 0.1, 0.9],
            'action': {'func': 'sloop_object_search.oopomdp.domain.action.basic_discrete_moves2d',
                       'params': {'h_rotation': 45.0,
                                  'step_size': 1,
                                  "yaxis": "up"}},
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]}},
        'targets': ['book'],
        'misc': {
            'visual': {'res': 25},
        }
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



######### THE FOLLOWING IS USED FOR HIERARCHICAL SEARCH TEST #########
CONFIG_HIER = {
    "object_locations": {
        "book": [0.0, 2.0, 0.1],
        "cup": [1.2, 0.1, 0.5]
    },

    "agent_config": {
        "agent_class": "MosAgentTopo2D",
        "agent_type": "hierarchical",  # 'hierarchical' or 'local'
        "belief": {},
        "robot": {
            "id": "robot0",
            "no_look": True,
            # "sensors": [{"name": "camera",
            #              "params": {"fov": 61,
            #                         "far": 1.75,
            #                         "near": 0.2,
            #                         "occlusion_enabled": True}}],

            "detectors": {"book": {"class": 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   "params": {"sensor": "camera",
                                              "quality": [1e5, 0.1]}}},

            "sensors": [{"name": 'fan',
                         'params': {'fov': 61,
                                    'min_range': 0.2,
                                    'max_range': 1.75}}],
            'detectors': {'book': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                                   'params': {"sensor": "fan",
                                              "quality": [1e5, 0.05]}}},
            "action": {"topo": {}},
            'color': [0.9, 0.1, 0.1, 0.9],
            #### Below are specific to hierarchical type agents ####
            "sensors_local": [{'name': 'camera',
                               'params': {'fov': 61,
                                          'far': 1.75,
                                          'near': 0.2,
                                          'occlusion_enabled': True}}],
            "detectors_local": {'book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                         'params': {"sensor": "camera",
                                                    "quality": [1e5, 0.1]}},
                                # 'cup': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                #          'params': {"sensor": "camera",
                                #                     "quality": [1e5, 0.1]}}},
                                },
            "action_local": {"topo": {}}
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
        'misc': {
            'visual': {'res': 25},
        }
    },

    "task_config": {
        "max_steps": 100
    },

    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "exploration_const": 1000,
            "max_depth": 8,
            "num_sims": 400,
            "show_progress": True
        }
    }
}


#### SET WHICH CONFIG TO USE ###
CONFIG = CONFIG_HIER

import yaml
def main():
    with open("config_simple_sim_lab121_lidar.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
