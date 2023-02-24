######### SEARCH TRIAL CONFIGURATIONS ######
TARGET_OBJECTS = ["Cat", "Lyzol", ]  #"ToyPlane", "Pringles"]

SEARCH_REGION_3D = {"res": 0.15,
                    "octree_size": 32,
                    "region_size_x": 4.0,
                    "region_size_y": 5.0,
                    "region_size_z": 2.5,
                    "center_x": 0.7050081253051758,
                    "center_y": 4.158568382263184,
                    "center_z": 0.0,
                    "center_qx": 0.0,
                    "center_qy": 0.0,
                    "center_qz": 0.9993078786023883,
                    "center_qw": 0.03719897529763721,
                    "debug": False}


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
              # # the 3D box within which samples of viewpoint positions will be drawn.
              # 'sample_space': {
              #       "center_x": 1.5050081253051758,
              #       "center_y": 4.158568382263184,
              #       "center_z": 0.0,
              #     "size_x": 2.00,
              #     "size_y": 1.50,
              #     "size_z": 2.00
              # },
              '3d_proj_2d': {'layout_cut': 0.4,
                             'floor_cut': 0.15,
                             'brush_size': 0.2,
                             'inflation': 0.5}}

LOCAL_ACTION = {'topo': LOCAL_TOPO,
                'policy': {'cost_scaling_factor': 1.0}}

LOCAL_REACHABLE = {"min_height": 1.3,
                   "max_height": 1.6}

LOCAL_PLANNER_CONFIG = {"planner": "pomdp_py.POUCT",
                        "planner_params": {
                            "exploration_const": 1000,
                            "max_depth": 6,
                            "num_sims": 400,
                            "show_progress": True}}

######### DETECTORS AND OBJECTS ######
LOCAL_DETECTORS = {
    'Cat': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
            'params': {"sensor": "hand_camera",
                       "quality": [1e7, 0.2]}},
    'Bowl': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
             'params': {"sensor": "hand_camera",
                        "quality": [1e7, 0.2]}},
    'Columbia Book': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                      'params': {"sensor": "hand_camera",
                                 "quality": [1e7, 0.2]}},
    'ToyPlane': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e7, 0.2]}},
    'Lyzol': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e7, 0.2]}},
    'BlackPump': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e7, 0.2]}},
    'Pringles': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                 'params': {"sensor": "hand_camera",
                            "quality": [1e7, 0.2]}},
    'Robot Book': {'class': 'genmos_object_search.oopomdp.FrustumVoxelAlphaBeta',
                   'params': {"sensor": "hand_camera",
                            "quality": [1e7, 0.2]}},
}

OBJECTS = {
    'Cat': {'class': 'Cat',
            'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
            'color': [0.67, 0.61, 0.15, 0.8],
            'viz_type': 'cube'},

    'Bowl': {'class': 'Bowl',
             'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
             'color': [0.82, 0.61, 0.01, 0.8],
             'viz_type': 'cube'},

    'Columbia Book': {'class': 'Columbia Book',
                      'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                      'color': [0.01, 0.5, 0.81, 0.8],
                      'viz_type': 'cube'},

    'ToyPlane': {'class': 'ToyPlane',
                 'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                 'color': [0.01, 0.76, 0.85, 0.8],
                 'viz_type': 'cube'},

    'Lyzol': {'class': 'Lyzol',
              'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
              'color': [0.8, 0.75, 0.54, 0.8],
              'viz_type': 'cube'},

    'BlackPump': {'class': 'BlackPump',
                  'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                  'color': [0.34, 0.34, 0.35, 0.8],
                  'viz_type': 'cube'},

    'Pringles': {'class': 'Pringles',
                 'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
                 'color': [0.25, 0.72, 0.01, 0.8],
                 'viz_type': 'cube'},

    'Robot Book': {'class': 'Robot Book',
                 'transition': {'class': 'genmos_object_search.oopomdp.StaticObjectTransitionModel'},
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
            'detectors': make_detectors("local", *TARGET_OBJECTS),
            'action': LOCAL_ACTION,
            "reachable": LOCAL_REACHABLE,
        },
        'objects': make_objects(*TARGET_OBJECTS),
        'targets': TARGET_OBJECTS
    },

    "task_config": {
        "max_steps": 200
    },

    "planner_config": LOCAL_PLANNER_CONFIG
}

CONFIG = CONFIG_LOCAL

import yaml
import os
def main():
    with open("{}.yaml".format(os.path.basename(__file__).split(".")[0]), "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
