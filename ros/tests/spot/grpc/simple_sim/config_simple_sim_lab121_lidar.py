RANDOM_POSITIONS_BOOK =\
    [[-0.027, 0.152, -0.177],
     [-0.718, 2.571, 0.560],
     [-0.642, 1.594, 0.077],
     [-0.001, -0.173, 0.189],
     [-0.515, -0.465, 1.065],
     [-0.736, 2.586, 0.699],
     [-0.354, 2.875, 0.194],
     [-0.140, 0.789, -0.388],
     [-0.236, 3.174, 1.399],
     [-0.512, -0.356, 0.463],
     [0.528, 3.188, -0.313],
     [0.128, 2.049, -0.252],
     [-0.611, 0.978, 0.025],
     [0.266, 2.366, -0.134],
     [-0.709, 2.667, -0.006],
     [0.515, 1.304, -0.268],
     [-0.158, 1.905, -0.138],
     [0.994, -0.287, -0.317],
     [-0.019, -0.205, -0.085],
     [-0.121, 0.389, -0.379],
     [-0.346, 0.788, -0.083],
     [-0.516, 1.898, -0.164],
     [-0.497, -0.622, 0.555],
     [-0.535, 1.969, 0.157],
     [-0.786, -0.199, -0.280],
     [-0.681, 2.270, 0.780],
     [-0.562, 0.409, 0.516],
     [0.962, 1.258, -0.351],
     [2.349, 2.585, -0.390],
     [-0.276, 1.008, 0.357]]

RANDOM_POSITIONS_CUP =\
    [[-0.587, 1.017, 0.844],
     [0.214, 3.181, -0.188],
     [-0.755, 2.802, 0.229],
     [-0.226, 3.170, 0.965],
     [-0.428, 0.698, -0.398],
     [-0.241, 1.560, -0.352],
     [-0.336, 1.572, -0.400],
     [-0.638, 1.207, 0.518],
     [-0.626, 1.220, 0.919],
     [2.891, -0.606, -0.371],
     [-0.587, 1.680, 0.482],
     [-0.572, 0.301, 1.020],
     [0.717, 3.058, -0.371],
     [-0.747, 0.980, 0.082],
     [0.004, -0.066, 0.249],
     [-0.206, 3.016, 0.009],
     [-0.548, 0.114, 0.916],
     [-0.311, -0.075, 0.450],
     [-0.117, 2.676, -0.161],
     [-0.486, -0.647, 1.141],
     [-0.644, 1.372, 0.068],
     [-0.598, 1.638, 0.080],
     [-0.591, 0.280, 1.241],
     [-0.426, 0.620, 0.613],
     [-0.685, 2.175, 0.841],
     [-0.617, 0.998, 0.719],
     [-0.643, 1.916, 1.189],
     [-0.101, 1.896, 0.084],
     [-0.483, 0.347, 0.153],
     [-0.680, 2.194, 0.777]]

######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
LOCAL_TOPO = {'num_nodes': 10,
              'pos_importance_thres': 0.01,
              'resample_thres': 0.4,
              'sep': 0.75,
              'debug': False,
              'res_buf': 4,
              '3d_proj_2d': {'layout_cut': 0.4,
                             'floor_cut': 0.15,
                             'brush_size': 0.2,
                             'inflation': 0.1}}

CONFIG_LOCAL = {
    "object_locations": {
        "book": RANDOM_POSITIONS_BOOK,
        "cup": RANDOM_POSITIONS_CUP
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
                'params': {'fov': 60,
                           'far': 2.50,
                           'near': 0.2,
                           'occlusion_enabled': True}
            }],
            'detectors': {'book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e8, 0.1]}},
                          'cup': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e8, 0.1]}}},
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
        "max_time": 180
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
                         'params': {'fov': 60,
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
