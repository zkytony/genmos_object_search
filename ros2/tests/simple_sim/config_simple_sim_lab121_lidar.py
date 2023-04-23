RANDOM_POSITIONS_BOOK =\
    [[-0.44932237, -0.29711649, 0.4289201],
     [-0.17160062, 0.59915698, -0.1856396],
     [-0.52430302, -0.08935844, 0.67364234],
     [-0.5106892, -0.63244444, -0.0881151],
     [-0.53119594, 0.67405683, 0.20810561],
     [-0.42141247, 1.07187784, 0.08253412],
     [-0.35860109, 0.54058391, 0.63191831],
     [-0.53340948, 0.2539537, 0.80852979],
     [-0.63593853, 1.73557889, 0.92626876],
     [-0.51800716, -0.03557291, 1.44831145],
     [-0.57193404, 0.74773407, 0.79149574],
     [-0.60966909, 1.31032598, 0.04862576],
     [-0.42486256, 1.12093222, 0.33885267],
     [-0.44203654, 1.74377823, 0.60913336],
     [-0.40380728, 0.81785244, 0.62644839],
     [-0.60061222, 1.11406958, 0.05310385],
     [-0.34810784, -0.02355452, 0.27944404],
     [-0.55510938, 2.23833895, 1.22022676],
     [-0.64735264, 1.95332944, 0.92745602],
     [-0.35029355, 0.89687872, 0.10148292],
     [-0.63999057, 2.07852006, -0.02671536],
     [-0.28638339, 1.79643607, -0.04134189],
     [0.03305916, -0.03869896, 0.28033626],
     [-0.60037899, 2.00861096, 0.17775048],
     [-0.14880879, 0.613729, -0.05003705],
     [-0.63454604, 1.68140185, 1.07825267],
     [-0.43311819, 0.33586475, 0.35854083],
     [-0.60009903, 1.67628813, 0.18606916],
     [-0.30004144, 1.8429389, 0.32186273],
     [-0.39811662, 1.77725625, 0.46473926]]

RANDOM_POSITIONS_CUP =\
    [[-0.37163266, 1.10503125, 0.54144536],
     [-0.41113988, 1.78431857, -0.03150767],
     [-0.05718368, 0.13527155, 0.35534877],
     [-0.05482426, 0.12062754, 0.22286218],
     [-0.01737663, -0.23077844, -0.04446195],
     [-0.58806896, 1.88674104, 0.47374803],
     [-0.58326238, 2.32824826, 1.22928655],
     [-0.58393043, 0.89893281, 0.05769631],
     [-0.56232703, 0.68285513, 0.79277098],
     [-0.52172965, 1.82076716, 0.70525491],
     [-0.6820944, 2.22845554, 0.47430223],
     [-0.52706945, 1.21572673, 0.341562],
     [-0.53492427, 0.1114245, 0.3662475],
     [-0.55002385, 0.76126009, 0.0814532],
     [-0.58919895, 1.03180766, 1.08247006],
     [-0.62905031, 1.52860689, 0.04255061],
     [-0.55370635, 2.08828521, 1.21649265],
     [-0.59534574, 1.18676353, 0.6337049],
     [-0.51338124, -0.30121848, 1.15286267],
     [-0.67201501, 2.38510442, 0.31926605],
     [-0.56255966, 1.32066023, 0.11301935],
     [-0.44495863, 1.67291117, 0.61035365],
     [-0.59637505, 2.0579021, 0.0774008],
     [-0.00983155, -0.11246746, 0.22648418],
     [-0.54387736, -0.03562904, 0.52112418],
     [-0.53482264, 0.24248599, 1.26817477],
     [0.06722162, 0.41785207, -0.03225271],
     [-0.49080002, -0.58875245, 0.69667655],
     [-0.26105437, 1.53802776, 0.04781055],
     [-0.65922832, 2.10052156, -0.12615938]]

######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
LOCAL_TOPO = {'num_nodes': 10,
              'pos_importance_thres': 0.01,
              'resample_thres': 0.4,
              'sep': 0.75,
              'debug': False,
              'res_buf': 4,
              '3d_proj_2d': {'layout_cut': 0.4,
                             'floor_cut': 0.1,
                             'brush_size': 0.2,
                             'inflation': 0.05}}

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
                   "init_params": {"num_samples": 3000,
                                   "prior_from_occupancy": True,
                                   "occupancy_height_thres": 0.0,
                                   "occupancy_blow_up_res": 4,
                                   "occupancy_fill_height": True}},
        "search_region": {},
        'robot': {
            'id': 'robot0',
            'no_look': True,
            'sensors': [{
                'name': 'camera',
                'params': {'fov': 60,
                           'far': 2.00,
                           'near': 0.2,
                           'occlusion_enabled': True}
            }],
            'detectors': {'book': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e8, 0.1]}},
                          'cup': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e8, 0.1]}}},
            'action': {'topo': LOCAL_TOPO,
                       # Below is not relevant if you are using Topo3D
                       'func': 'genmos_object_search.oopomdp.domain.action.basic_discrete_moves3d',
                       'params': {'step_size': 0.2,
                                  'rotation': 90.0,
                                  'scheme': 'axis'}},
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                             'color': [0.4, 0.7, 0.3, 0.8],
                             'viz_type': 'cube',
                             'sizes': [0.14, 0.08, 0.10]},
                    'cup': {'class': 'cup',
                            'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                            'color': [0.89, 0.6, 0.05, 0.8],
                            'sizes': [0.12, 0.12, 0.12],
                            'viz_type': 'cube'}},
        'targets': ['book', 'cup'],
    },

    "task_config": {
        "max_steps": 100,
        "max_time": 180
    },

    "planner_config":  {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "exploration_const": 1000,
            "max_depth": 8,
            "num_sims": 200,
            "show_progress": True
        }
        ####################################
        # some other possibilities for planner_config:
        #
        #     "planner": "genmos_object_search.oopomdp.planner.random.RandomPlanner",
        #     "planner_params": {}
        #  or,
        #
        #     "planner": "genmos_object_search.oopomdp.planner.greedy.GreedyPlanner",
        #     "planner_params": {}
        #     }
        #
        #####################################
    },

    "ros2": {
        "detection_class_names": ["book", "cup"]
    }
}



######### THE FOLLOWING IS USED FOR 2D LOCAL SEARCH TEST #########
CONFIG_LOCAL2D = {
    "object_locations": {
        "book": [[0.0, 2.0, 0.1]],
        "cup": [[1.2, 0.1, 0.5]]
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
            'detectors': {'book': {'class': 'genmos_object_search.oopomdp.FanModelAlphaBeta',
                                   'params': {"sensor": "fan",
                                              "quality": [1e5, 0.05]}}},
            'color': [0.9, 0.1, 0.1, 0.9],
            'action': {'func': 'genmos_object_search.oopomdp.domain.action.basic_discrete_moves2d',
                       'params': {'h_rotation': 45.0,
                                  'step_size': 1,
                                  "yaxis": "up"}},
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
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
    },

    "ros2": {
        "detection_class_names": ["book", "cup"]
    }
}



######### THE FOLLOWING IS USED FOR TOPO2D SEARCH TEST #########
CONFIG_TOPO2D = {
    "object_locations": {
        "book": [[0.0, 2.0, 0.1]],
        "cup": [[1.2, 0.1, 0.5]]
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
            'detectors': {'book': {'class': 'genmos_object_search.oopomdp.FanModelAlphaBeta',
                                   'params': {"sensor": "fan",
                                              "quality": [1e5, 0.05]}}},
            "action": {"topo": {}},
            'color': [0.9, 0.1, 0.1, 0.9],
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
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
    },

    "ros2": {
        "detection_class_names": ["book", "cup"]
    }
}


######### THE FOLLOWING IS USED FOR HIERARCHICAL SEARCH TEST #########
CONFIG_HIER = {
    "object_locations": {
        "book": [[0.0, 2.0, 0.1]],
        "cup": [[1.2, 0.1, 0.5]]
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
            "detectors": {"book": {"class": 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   "params": {"sensor": "camera",
                                              "quality": [1e5, 0.1]}}},

            "sensors": [{"name": 'fan',
                         'params': {'fov': 61,
                                    'min_range': 0.2,
                                    'max_range': 1.75}}],
            'detectors': {'book': {'class': 'genmos_object_search.oopomdp.FanModelAlphaBeta',
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
            "detectors_local": {'book': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                         'params': {"sensor": "camera",
                                                    "quality": [1e5, 0.1]}}},
            "action_local": {"topo": {}}
        },
        'objects': {'book': {'class': 'book',
                             'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
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
        "planner": "genmos_object_search.oopomdp.planner.hier.HierPlanner",
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
    },

    "ros2": {
        "detection_class_names": ["book", "cup"]
    }
}

#### SET WHICH CONFIG TO USE ###
# CONFIG = CONFIG_LOCAL  # for 3D local search
# CONFIG = CONFIG_TOPO2D  # for 2D local search
CONFIG = CONFIG_HIER

GROUNDTRUTH_PRIOR = False
OBJLOC_INDEX = 0  # this must match the objloc_index field in the SimpleSimEnv object.
if GROUNDTRUTH_PRIOR:
    CONFIG["agent_config"]["belief"]["prior"] = {}
    for objid in CONFIG["agent_config"]["targets"]:
        CONFIG["agent_config"]["belief"]["prior"][objid]\
            = [[CONFIG["object_locations"][objid][OBJLOC_INDEX], 0.99]]

import yaml
def main():
    with open("config_simple_sim_lab121_lidar.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
