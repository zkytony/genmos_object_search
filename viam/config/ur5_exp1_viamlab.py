######### COMMON CONFIGURATIONS ######
LOCAL_BELIEF = {"visible_volume_params": {"num_rays": 150,
                                          "step_size": 0.2,
                                          "voxel_res": 2},
                "init_params": {"num_samples": 3000,
                                "prior_from_occupancy": True,
                                "occupancy_height_thres": 0.2,
                                "occupancy_blow_up_res": 4,
                                "occupancy_fill_height": True}}

SEARCH_REGION_3D = {"res": 0.1,
                    "octree_size": 32,
                    "region_size_x": 2.5,
                    "region_size_y": 2.5,
                    "region_size_z": 1.5,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0}

# Official spot spec: RGB: 60.2deg x 46.4deg; Depth: 55.9deg x 44deg
GRIPPER_CAMERA = {'name': 'gripper_camera',
                  'params': {'fov': 53,
                             'far': 1.5,
                             'near': 0.2,
                             'aspect_ratio': 0.7,
                             'occlusion_enabled': True}}

LOCAL_TOPO = {'num_nodes': 5,
              'pos_importance_thres': 0.01,
              'sep': 0.25,
              'debug': False,
              'resample_thres': 0.4,
              '3d_proj_2d': {'layout_cut': 0.4,
                             'floor_cut': 0.15,
                             'brush_size': 0.2,
                             'inflation': 0.5}}

LOCAL_ACTION = {'topo': LOCAL_TOPO,
                'policy': {'cost_scaling_factor': 1.0}}

LOCAL_REACHABLE = {"min_height": 1.0,
                   "max_height": 1.3}

LOCAL_PLANNER_CONFIG = {"planner": "pomdp_py.POUCT",
                        "planner_params": {
                            "exploration_const": 1000,
                            "max_depth": 6,
                            "num_sims": 400,
                            "show_progress": True}}

######### DETECTORS AND OBJECTS ######
LOCAL_DETECTORS = {
    'Chair': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
            'params': {"sensor": "gripper_camera",
                       "quality": [1e7, 0.2]}},
}

OBJECTS = {
    'Chair': {'class': 'Chair',
            'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
            'color': [0.67, 0.61, 0.15, 0.8],
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
target_objects = ["Chair"]
CONFIG_LOCAL = {
    "agent_config": {
        "agent_class": "MosAgentTopo3D",
        "agent_type": "local",
        'belief': LOCAL_BELIEF,
        "search_region": {"3d": SEARCH_REGION_3D},
        'robot': {
            'id': 'robot0',
            'no_look': True,
            'sensors': [GRIPPER_CAMERA],
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


CONFIG = CONFIG_LOCAL

import yaml
def main():
    with open("ur5_exp1_viamlab.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
