DETECTORS = {
    'Cat': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e6, 0.1]}},
}

OBJECTS = {
    {'Cat': {'class': 'Cat',
             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'},
             'color': [0.4, 0.7, 0.3, 0.8],
             'viz_type': 'cube',
             'sizes': [0.14, 0.08, 0.10]}}
}

def make_objects(*objects):
    result = {}
    for obj in objects:
        result[obj] = OBJECTS[obj]
    return result

def make_detectors(*objects):
    result = {}
    for obj in objects:
        result[obj] = DETECTORS[obj]
    return result

######### THE FOLLOWING IS USED FOR LOCAL SEARCH TEST #########
CONFIG_LOCAL = {
    "agent_config": {
        "agent_class": "MosAgentTopo3D",
        "agent_type": "local",
        'belief': {"visible_volume_params": {"num_rays": 150,
                                             "step_size": 0.4,
                                             "voxel_res": 2},
                   "init_params": {"num_samples": 3000,
                                   "prior_from_occupancy": True}},
        "search_region": {"3d": {"res": 0.07,
                                 "region_size_x": 2.5,
                                 "region_size_y": 2.5,
                                 "region_size_z": 1.8}},
        'robot': {
            'id': 'robot0',
            'no_look': True,
            'sensors': [{
                'name': 'camera',
                'params': {'fov': 50,
                           'far': 1.5,
                           'near': 0.2,
                           'aspect_ratio': 0.5,
                           'occlusion_enabled': True}
            }],
            'detectors': make_detectors("Cat"),
            'action': {'topo': {'num_nodes': 10,
                                'pos_importance_thres': 0.01,
                                'sep': 0.75,
                                'debug': False,
                                '3d_proj_2d': {'layout_cut': 0.4,
                                               'floor_cut': 0.15,
                                               'brush_size': 0.2,
                                               'inflation': 0.25}},
                       'policy': {'cost_scaling_factor': 1.0}},
            "reachable": {
                "min_height": 0.5,
                "max_height": 1.2
            }
        },
        'objects': make_objects("Cat"),
        'targets': ['Cat'],
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
}

CONFIG = CONFIG_LOCAL

import yaml
def main():
    with open("spot_exp1_local_lab121.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

if __name__ == "__main__":
    main()
