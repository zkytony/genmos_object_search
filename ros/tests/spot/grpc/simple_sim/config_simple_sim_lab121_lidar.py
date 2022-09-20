######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
LOCAL_TOPO = {'num_nodes': 10,
              'pos_importance_thres': 0.01,
              'resample_thres': 0.4,
              'sep': 0.75,
              'debug': False,
              '3d_proj_2d': None,
              'res_buf': 2}

CONFIG_LOCAL = {
    "object_locations": {
        "book": [[0.0, 2.0, 0.1], [1.0, 2.0, 1.1], [1.384, -0.561, -0.128],
                 [2.168, 0.454, 0.73], [-0.249, -0.198, 1.177], [0.354, 1.18,
                 0.975], [2.016, 1.659, -0.135], [1.246, 0.822, 0.82], [1.658,
                 -0.5, -0.011], [1.78, 2.16, 1.356], [2.234, -0.111, 0.397],
                 [1.694, 1.253, 1.117], [1.753, -0.168, -0.375], [-0.578, 1.865,
                 0.648], [1.24, -0.575, -0.612], [1.394, 1.688, -0.585], [2.082,
                 0.151, 0.719], [0.048, 2.176, 0.398], [0.91, -0.149, 0.814],
                 [0.19, 0.061, -0.337], [0.598, 0.728, 0.059], [1.124, 1.821,
                 1.362], [-0.503, 0.651, -0.411], [-0.725, 2.13, 0.158],
                 [-0.731, -0.146, 0.671], [1.265, 1.978, 0.859], [1.84, 2.04,
                 1.483], [1.09, -0.372, 0.518], [1.352, 0.049, 0.728], [2.01,
                 -0.398, 0.411], [-0.663, 1.205, -0.629], [1.191, 0.92, 0.798],
                 [0.167, -0.2, -0.431], [0.261, 1.552, 1.317], [1.284, 0.774,
                 0.985], [2.284, 1.541, 1.02], [2.091, 2.01, 0.891], [1.451,
                 1.642, -0.319], [0.108, 2.153, -0.434], [0.221, 1.337, 0.641],
                 [0.023, 1.21, 1.497], [1.629, 2.112, 0.362], [-0.549, -0.425,
                 0.234], [0.77, 0.854, -0.397], [-0.275, 1.24, -0.188], [-0.314,
                 -0.176, 0.181], [2.118, 1.594, -0.573], [1.294, 1.499, -0.269],
                 [-0.589, -0.634, 0.19], [2.054, 1.906, -0.237], [-0.464, 2.12,
                 -0.693], [1.516, -0.263, 1.517], [1.309, 0.668, 0.442], [0.582,
                 -0.288, 0.423], [1.502, 0.79, 0.303], [-0.098, 1.95, -0.198],
                 [2.133, -0.45, 0.878], [1.909, 0.513, 1.213], [2.242, 2.102,
                 0.213], [0.321, 0.56, 1.129], [0.177, 2.263, 0.736], [0.648,
                 -0.659, 0.717], [2.097, 2.071, -0.552], [-0.622, -0.214,
                 -0.694], [-0.171, 0.065, 0.58], [1.107, 1.932, -0.347], [0.997,
                 -0.37, 0.901], [-0.248, -0.289, -0.408], [1.664, 1.516, 0.257],
                 [1.174, 1.838, 0.861], [1.07, 1.038, 0.838], [0.737, 1.348,
                 1.16], [0.098, 0.957, -0.614], [0.754, 0.643, 1.161], [1.077,
                 0.925, 0.544], [0.061, 1.684, 0.615], [1.465, 1.605, -0.616],
                 [0.396, -0.213, 1.023], [2.253, 1.173, 0.78], [1.922, 0.59,
                 1.542], [1.159, 0.631, -0.276], [-0.54, 1.315, -0.455],
                 [-0.243, 1.302, 0.459], [-0.245, 2.308, 1.575], [2.066, -0.423,
                 0.917], [1.145, 0.923, -0.328], [1.284, 0.314, 0.056], [0.59,
                 0.581, -0.694], [0.586, 0.866, 0.993], [0.972, 0.409, 0.236],
                 [1.283, 0.296, 0.59], [0.975, 0.952, 0.999], [2.206, 2.035,
                 -0.586], [-0.736, -0.674, 1.422], [1.85, -0.624, 0.564],
                 [1.937, -0.312, 1.324], [2.061, 2.386, -0.559], [-0.12, -0.611,
                 0.018], [-0.185, 1.512, 0.449], [2.249, 1.735, 1.342], [2.222,
                 0.579, 0.86], [1.668, 1.72, 0.699]],

        "cup": [[1.35, 0.1, 0.4], [1.35, 1.5, 0.1], [2.171, -0.47, 1.416], [-0.192, 2.296,
                -0.678], [-0.76, 1.125, 0.164], [-0.511, 0.933, 1.237], [0.541, 2.205, 0.313],
                [-0.306, -0.422, 0.797], [2.023, 1.995, 0.604], [0.292, 1.174, -0.475], [0.497,
                -0.478, 0.104], [0.289, 2.192, 0.978], [-0.44, 0.216, 0.385], [1.206, -0.503,
                0.38], [0.656, 0.206, 1.328], [-0.074, -0.509, 0.367], [-0.333, 1.985, -0.419],
                [-0.432, -0.495, -0.151 ], [1.492, -0.101, 0.372], [0.924, 0.255, 1.055],
                [0.403, -0.298, 1.096], [-0.595, 1.032, -0.634], [2.119, 1.309, 0.809], [-0.712,
                1.058, 0.053], [1.586, 2.303, 0.708], [2.136, 0.0, -0.378], [1.434, 0.489,
                1.055], [0.094, 0.003, 0.612], [-0.578, 1.663, 0.827], [1.167, -0.664, 0.609],
                [0.186, 1.0, -0.154], [0.131, 0.355, -0.417], [0.529, 1.824, 0.083], [0.878,
                1.787, 0.81], [0.633, 1.503, 1.061], [-0.652, 0.837, 0.012], [-0.195, -0.041,
                0.449], [1.6, -0.452, -0.092], [-0.18, 1.256, 0.676], [1.554, 1.389, 0.782],
                [2.26, 0.373, 1.104], [1.477, -0.149, 0.771], [2.254, 2.145, 1.437], [2.014,
                1.951, 0.532], [0.075, -0.505, 1.402], [2.253, -0.659, 0.691], [0.224, 0.367,
                1.226], [1.782, 1.937, 0.788], [0.935, -0.696, -0.207], [1.247, -0.081, -0.549],
                [2.064, 0.625, -0.663], [0.271, 1.453, 0.741], [0.87, 2.301, 0.495], [0.741,
                0.565, 0.053], [-0.001, 1.704, 0.632], [0.482, -0.146, 0.696], [-0.314, 1.952,
                -0.248], [0.013, -0.251, -0.661], [-0.135 , 0.249, 0.325], [1.906, 1.768,
                1.415], [0.969, 2.071, 1.099], [1.751, 0.889, 0.477], [-0.208, -0.676, -0.107],
                [1.422, 0.466, 0.411], [2.06, 1.328, 1.376], [-0.7, 1.554, 1.019], [0.883, 0.021, 1.148], [0.13, -0.448, 0.092], [1.644, 0.364, -0.584], [-0.776, 2.188,
                -0.194], [2.242, 0.447, 0.569], [0.55, 1.4, 1.539], [0.632, 1.291, -0.33],
                [-0.036, 1.545, 0.36], [1.058, 2.021, 0.412], [2.012, 0.909, -0.519], [1.657,
                1.408, 1.542], [-0.541, -0.622, 0.97], [-0.588, 0.607, 0.82], [0.312, 0.781,
                1.442], [1.314, 1.427, 0.381], [-0.105, 0.194, 1.52], [-0.44, 1.781, 0.628] ,
                [0.929, -0.249, 1.598], [-0.471, 1.228, 1.547], [1.306, 0.201, -0.657], [1.714,
                -0.692, -0.423], [2.101, 1.493, -0.273], [0.782, 2.205, 1.511], [-0.76, 1.635,
                -0.48], [1.464, 2.266, 1.194] , [-0.118, 1.686, 1.559], [1.561, 1.009, -0.633],
                [2.204, 2.098, -0.18], [0.994, 1.725, -0.191], [-0.259, 1.931, 0.775], [0.199,
                2.361, -0.125], [1.591, 0.702, -0.671], [1.815, 1.721, 0.73], [-0.713, 0.036,
                -0.121], [1.721, -0.113, 0.548], [1.544, 0.582, 0.981]]
    },

    "agent_config": {
        "agent_class": "MosAgentTopo3D", #"MosAgentBasic3D",
        "agent_type": "local",
        'belief': {"visible_volume_params": {"num_rays": 150,
                                             "step_size": 0.4,
                                             "voxel_res": 2},
                   "init_params": {"num_samples": 3000}},
        "search_region": {},
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
                                              "quality": [1e6, 0.1]}},
                          'cup': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e6, 0.1]}}},
            'action': {'topo': LOCAL_TOPO,
                       # Below is not relevant if you are using Topo3D
                       'func': 'sloop_object_search.oopomdp.domain.action.basic_discrete_moves3d',
                       'params': {'step_size': 0.2,
                                  'rotation': 90.0,
                                  'scheme': 'axis'}},
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]},
                    'cup': {'class': 'cup',
                            'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                            'color': [0.89, 0.6, 0.05, 0.8],
                            'sizes': [0.12, 0.12, 0.12],
                            'viz_type': 'cube'}},
        'targets': ['book', 'cup'],
    },

    "task_config": {
        "max_steps": 100,
        "max_time": 120
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
        "agent_type": "local",
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



######### THE FOLLOWING IS USED FOR TOPO2D SEARCH TEST #########
CONFIG_TOPO2D = {
    "object_locations": {
        "book": [0.0, 2.0, 0.1],
        "cup": [1.2, 0.1, 0.5]
    },

    "agent_config": {
        "agent_class": "MosAgentTopo2D",
        "agent_type": "local",
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
            "action": {"topo": {}},
            'color': [0.9, 0.1, 0.1, 0.9],
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
            "num_sims": 400,
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
        "belief_local": {"visible_volume_params": {"num_rays": 150,
                                                   "step_size": 0.4,
                                                   "voxel_res": 2}},
        "robot": {
            "id": "robot0",
            "no_look": True,
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
                                                    "quality": [1e5, 0.1]}}},
            "action_local": {"topo": {}}
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]}},
        'targets': ['book'],
        'misc': {
            'visual': {'res': 15},
        }
    },

    "task_config": {
        "max_steps": 100,
        "max_time": 5
    },

    "planner_config": {
        "planner": "sloop_object_search.oopomdp.planner.hier.HierPlanner",
        "planner_params": {
            "global": {
                "exploration_const": 1000,
                "max_depth": 8,
                "num_sims": 400,
                "show_progress": True
            },
            "local": {
                "exploration_const": 1000,
                "max_depth": 8,
                "num_sims": 400,
                "show_progress": True
            }
        }
    }
}

#### SET WHICH CONFIG TO USE ###
CONFIG = CONFIG_LOCAL



import yaml
def main():
    with open("config_simple_sim_lab121_lidar.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
