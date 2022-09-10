TEST_CONFIG = {
    'agent_config': {
        'action': {'h_rotation': 45.0},
        'agent_class': 'MosAgentBasic3D',
        'agent_type': 'local',
        'belief': {"visible_volume_params": {"num_rays": 150,
                                             "step_size": 0.4,
                                             "voxel_res": 2},
                   "init_params": {"num_samples": 3000}},
        'detectable_objects': ['book'],
        'foref_model_map_name': 'honolulu',
        'foref_models_dir': '/home/kaiyu/repo/robotdev/shared/ros/sloop_object_search/sloop_object_search/models',
        'no_look': True,
        'object_symbol_map': {'NovelBook': 'book'},
        'objects': {'book': {'class': 'book',
                             'color': [200, 100, 80],
                             'transition': {'class': 'sloop_object_search.oopomdp.StaticObjectTransitionModel'}}},
        'robot': {
            'sensors': [{
                'name': 'camera',
                'params': {'fov': 61,
                           'far': 1.5,
                           'near': 0.1,
                           'occlusion_enabled': True}
            }],
            'detectors': {'book': {'class': 'sloop_object_search.oopomdp.FrustumVoxelAlphaBeta',
                                   'params': {"sensor": "camera",
                                              "quality": [1e5, 0.1]}}},
                  'id': 'robot0',
                  'action': {'func': 'sloop_object_search.oopomdp.domain.action.basic_discrete_moves3d',
                             'params': {'step_size': 0.2,
                                        'rotation': 90.0,
                                        'scheme': 'axis'}}},
        'spacy_model': 'en_core_web_lg',
        'targets': ['book'],
        'topo_map_args': {'degree': [3, 5],
                          'node_coverage_radius': 3.0,
                          'num_place_samples': 10,
                          'resample_prob_thres': 0.25,
                          'seed': 1509,
                          'sep': 4.0},
        'topo_trans_args': {'h_angle_res': 45.0},
        'visualizer': 'sloop_object_search.oopomdp.agent.VizSloopMosTopo',
        'viz_params': {'init': {'res': 20},
                       'render': {'show_img_flip_horizontally': True}}},
    'planner_config': {'high_level_planner_args': {'exploration_const': 1000,
                                                   'max_depth': 20,
                                                   'num_sims': 600},
                       'local_search': {'planner': 'pomdp_py.POUCT',
                                        'planner_args': {'exploration_const': 1000,
                                                         'max_depth': 10,
                                                         'num_sims': 600}},
                       'plan_nav_actions': False,
                       'planner': 'sloop_object_search.oopomdp.planner.hier2d.HierarchicalPlanner'},
    'task_config': {'map_name': 'lab121_lidar', 'max_steps': 100}}
