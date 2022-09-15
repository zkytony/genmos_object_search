######### COMMON CONFIGURATIONS ######
LOCAL_BELIEF = {"visible_volume_params": {"num_rays": 150,
                                          "step_size": 0.4,
                                          "voxel_res": 2},
                "init_params": {"num_samples": 3000,
                                "prior_from_occupancy": False}}

GLOBAL_BELIEF = {"init_params": {"prior_from_occupancy": False}}

SEARCH_REGION_3D = {"res": 0.1,
                    "octree_size": 32,
                    "region_size_x": 3.0,
                    "region_size_y": 3.0,
                    "region_size_z": 1.5}

SEARCH_REGION_2D = {"res": 0.35,
                    "region_size": 10.0,
                    "layout_cut": 0.4,
                    "floor_cut": 0.15,
                    "brush_size": 0.5,
                    "expansion_width": 0.35}

# Official spot spec: RGB: 60.2deg x 46.4deg; Depth: 55.9deg x 44deg
HAND_CAMERA = {'name': 'hand_camera',
               'params': {'fov': 53,
                          'far': 1.5,
                          'near': 0.2,
                          'aspect_ratio': 0.7,
                          'occlusion_enabled': True}}

HAND_FAN = {"name": 'hand_fan',
            'params': {'fov': 50,
                       'min_range': 0.2,
                       'max_range': 1.5}}

LOCAL_TOPO = {'num_nodes': 10,
              'pos_importance_thres': 0.01,
              'sep': 0.75,
              'debug': False,
              'resample_thres': 0.4,
              '3d_proj_2d': {'layout_cut': 0.4,
                             'floor_cut': 0.15,
                             'brush_size': 0.2,
                             'inflation': 0.15}}

LOCAL_ACTION = {'topo': LOCAL_TOPO,
                'policy': {'cost_scaling_factor': 1.0}}

# Find cat under couch
# LOCAL_REACHABLE = {"min_height": 0.5,
#                    "max_height": 1.75}

# Another test
LOCAL_REACHABLE = {"min_height": 0.5,
                   "max_height": 1.3}

LOCAL_PLANNER_CONFIG = {"planner": "pomdp_py.POUCT",
                        "planner_params": {
                            "exploration_const": 1000,
                            "max_depth": 6,
                            "num_sims": 400,
                            "show_progress": True}}

GLOBAL_PLANNER_CONFIG = {"planner": "pomdp_py.POUCT",
                         "planner_params": {
                             "exploration_const": 1000,
                             "max_depth": 8,
                             "num_sims": 400,
                             "show_progress": True
                         }}

######### DETECTORS AND OBJECTS ######
LOCAL_DETECTORS = {
    'Cat': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
            'params': {"sensor": "hand_camera",
                       "quality": [1e8, 0.2]}},
    'Bowl': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
             'params': {"sensor": "hand_camera",
                        "quality": [1e8, 0.2]}},
    'Columbia Book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                      'params': {"sensor": "hand_camera",
                                 "quality": [1e8, 0.2]}},
    'ToyPlane': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e8, 0.2]}},
    'Lyzol': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e8, 0.2]}},
    'BlackPump': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e8, 0.2]}},
    'Pringles': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e8, 0.2]}},
    'Robot Book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                   'params': {"sensor": "hand_camera",
                            "quality": [1e8, 0.2]}},
}

GLOBAL_DETECTORS = {
    'Cat': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
            'params': {"sensor": "hand_fan",
                       "quality": [1e5, 0.4]}},
    'Bowl': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
             'params': {"sensor": "hand_fan",
                        "quality": [1e5, 0.4]}},
    'Columbia Book': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                      'params': {"sensor": "hand_fan",
                                 "quality": [1e5, 0.4]}},
    'ToyPlane': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                 'params': {"sensor": "hand_fan",
                            "quality": [1e5, 0.4]}},
    'Lyzol': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
              'params': {"sensor": "hand_fan",
                         "quality": [1e5, 0.4]}},
    'BlackPump': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                  'params': {"sensor": "hand_fan",
                             "quality": [1e5, 0.4]}},
    'Pringles': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                 'params': {"sensor": "hand_fan",
                            "quality": [1e5, 0.4]}},
    'Robot Book': {'class': 'sloop_object_search.oopomdp.FanModelAlphaBeta',
                 'params': {"sensor": "hand_fan",
                            "quality": [1e5, 0.4]}},
}

OBJECTS = {
    'Cat': {'class': 'Cat',
            'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
            'color': [0.67, 0.61, 0.15, 0.8],
            'viz_type': 'cube'},

    'Bowl': {'class': 'Bowl',
             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
             'color': [0.82, 0.61, 0.01, 0.8],
             'viz_type': 'cube'},

    'Columbia Book': {'class': 'Columbia Book',
                      'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                      'color': [0.01, 0.5, 0.81, 0.8],
                      'viz_type': 'cube'},

    'ToyPlane': {'class': 'ToyPlane',
                 'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                 'color': [0.01, 0.76, 0.85, 0.8],
                 'viz_type': 'cube'},

    'Lyzol': {'class': 'Lyzol',
              'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
              'color': [0.8, 0.75, 0.54, 0.8],
              'viz_type': 'cube'},

    'BlackPump': {'class': 'BlackPump',
                  'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                  'color': [0.34, 0.34, 0.35, 0.8],
                  'viz_type': 'cube'},

    'Pringles': {'class': 'Pringles',
                 'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                 'color': [0.25, 0.72, 0.01, 0.8],
                 'viz_type': 'cube'},

    'Robot Book': {'class': 'Robot Book',
                 'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
                 'color': [0.5, 0.28, 0.05, 0.8],
                 'viz_type': 'cube'},
}


def make_objects(*objects):
    result = {}
    for obj in objects:
        result[obj] = OBJECTS[obj]
    return result

def make_detectors(local_or_global, *objects):
    assert local_or_global in {"local", "global"}
    result = {}
    for obj in objects:
        if local_or_global == "local":
            result[obj] = LOCAL_DETECTORS[obj]
        else:
            result[obj] = GLOBAL_DETECTORS[obj]
    return result


######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
target_objects = ["Cat"]#, "Bowl", "Pringles"]#, "Robot Book"]
CONFIG_LOCAL = {
    "agent_config": {
        "agent_class": "MosAgentTopo3D",
        "agent_type": "local",
        'belief': LOCAL_BELIEF,
        "search_region": {"3d": SEARCH_REGION_3D},
        'robot': {
            'id': 'robot0',
            'no_look': True,
            'sensors': [HAND_CAMERA],
            'detectors': make_detectors("local", *target_objects),
            'action': LOCAL_ACTION,
            "reachable": LOCAL_REACHABLE,
        },
        'objects': make_objects(*target_objects),
        'targets': target_objects
    },

    "task_config": {
        "max_steps": 200
    },

    "planner_config": LOCAL_PLANNER_CONFIG
}

######### THE FOLLOWING IS USED FOR HIER SEARCH TEST #########
CONFIG_HIER = {
    "agent_config": {
        "agent_class": "MosAgentTopo2D",
        "agent_type": "hierarchical",  # 'hierarchical' or 'local'
        "belief": GLOBAL_BELIEF,
        "belief_local": LOCAL_BELIEF,
        "search_region": {"3d": SEARCH_REGION_3D,
                          "2d": SEARCH_REGION_2D},
        "robot": {
            "id": "robot0",
            "no_look": True,
            "detectors": make_detectors("global", *target_objects),
            "sensors": [HAND_FAN],
            "action": {"topo": {
                "inflation": 0.25,
                "debug": True
            }},
            #### Below are specific to hierarchical type agents ####
            "sensors_local": [HAND_CAMERA],
            "detectors_local": make_detectors("local", *target_objects),
            "action_local": LOCAL_ACTION,
            "reachable_local": LOCAL_REACHABLE,
        },
        'objects': make_objects(*target_objects),
        'targets': target_objects,
        'misc': {
            'visual': {'res': 5},
            'ros_visual': {'marker2d_z': -0.65}
        }
    },

    "task_config": {
        "max_steps": 100
    },

    "planner_config": {
        "planner": "sloop_object_search.oopomdp.planner.hier.HierPlanner",
        "planner_params": {
            "global": GLOBAL_PLANNER_CONFIG["planner_params"],
            "local": LOCAL_PLANNER_CONFIG["planner_params"]
        }
    }
}

CONFIG = CONFIG_HIER

import yaml
def main():
    with open("spot_exp3_atrium_cit1f.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
