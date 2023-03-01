# python test_simple_sim_env_local_search_3d.py groundtruth 32 config_simple_sim_lab121_lidar.yaml --planner pouct
# python test_simple_sim_env_local_search_3d.py occupancy 32 config_simple_sim_lab121_lidar.yaml --planner pouct
python test_simple_sim_env_local_search_3d.py occupancy 32 config_simple_sim_lab121_lidar.yaml --planner greedy
python test_simple_sim_env_local_search_3d.py occupancy 32 config_simple_sim_lab121_lidar.yaml --planner random

# python test_simple_sim_env_local_search_3d.py uniform 32 config_simple_sim_lab121_lidar.yaml --planner pouct

# python test_simple_sim_env_local_search_3d.py uniform 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner pouct
# python test_simple_sim_env_local_search_3d.py occupancy 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner pouct

# python test_simple_sim_env_local_search_3d.py groundtruth 32 config_simple_sim_lab121_lidar.yaml --planner greedy
# python test_simple_sim_env_local_search_3d.py uniform 32 config_simple_sim_lab121_lidar.yaml --planner greedy

# python test_simple_sim_env_local_search_3d.py uniform 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner greedy
# python test_simple_sim_env_local_search_3d.py occupancy 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner greedy

# python test_simple_sim_env_local_search_3d.py groundtruth 32 config_simple_sim_lab121_lidar.yaml --planner random
# python test_simple_sim_env_local_search_3d.py uniform 32 config_simple_sim_lab121_lidar.yaml --planner random

# python test_simple_sim_env_local_search_3d.py uniform 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner random
# python test_simple_sim_env_local_search_3d.py occupancy 16 config_simple_sim_lab121_lidar.yaml --res 0.2 --planner random
