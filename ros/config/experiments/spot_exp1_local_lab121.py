######### COMMON CONFIGURATIONS ######
LOCAL_BELIEF = {"visible_volume_params": {"num_rays": 150,
                                          "step_size": 0.2,
                                          "voxel_res": 2},
                "init_params": {"num_samples": 3000,
                                "prior_from_occupancy": True,
                                "occupancy_height_thres": 0.2,
                                "occupancy_blow_up_res": 4,
                                "occupancy_fill_height": True}}

GLOBAL_BELIEF = {"init_params": {"prior_from_occupancy": False}}

SEARCH_REGION_3D = {"res": 0.1,
                    "region_size_x": 3.0,
                    "region_size_y": 3.0,
                    "region_size_z": 1.5}

SEARCH_REGION_2D = {"res": 0.3,
                    "region_size": 4.0,
                    "layout_cut": 0.4,
                    "floor_cut": 0.15,
                    "brush_size": 0.5,
                    "expansion_width": 0.5}

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
                             'inflation': 0.5}}

LOCAL_ACTION = {'topo': LOCAL_TOPO,
                'policy': {'cost_scaling_factor': 1.0}}

LOCAL_REACHABLE = {"min_height": 0.5,
                   "max_height": 1.2}

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
                       "quality": [1e6, 0.2]}},
    'Bowl': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
             'params': {"sensor": "hand_camera",
                        "quality": [1e6, 0.2]}},
    'Columbia Book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                      'params': {"sensor": "hand_camera",
                                 "quality": [1e6, 0.2]}},
    'ToyPlane': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e6, 0.2]}},
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
                            "quality": [1e5, 0.4]}}
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
                 'viz_type': 'cube'}
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
            'detectors': make_detectors("local", "Cat", "ToyPlane", "Columbia Book"),
            'action': LOCAL_ACTION,
            "reachable": LOCAL_REACHABLE,
        },
        'objects': make_objects("Cat", "ToyPlane", "Columbia Book"),
        'targets': ['Cat', "ToyPlane", "Columbia Book"],
    },

    "task_config": {
        "max_steps": 100
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
            "detectors": make_detectors("global", "Cat"),
            "sensors": [HAND_FAN],
            "action": {"topo": {
                "inflation": 0.25,
                "debug": True
            }},
            #### Below are specific to hierarchical type agents ####
            "sensors_local": [HAND_CAMERA],
            "detectors_local": make_detectors("local", "Cat"),
            "action_local": LOCAL_ACTION,
            "reachable_local": LOCAL_REACHABLE,
        },
        'objects': make_objects("Cat"),
        'targets': ['Cat'],
        'misc': {
            'visual': {'res': 15},
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

CONFIG = CONFIG_LOCAL

import yaml
def main():
    with open("spot_exp1_local_lab121.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
