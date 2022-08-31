#!/usr/bin/env python
# Test the complete algorithm: hierarchical search.
# Performs hierarchical object search with 3D local
# search and 2D global search.
#
#
# To run the test, do the following IN ORDER:
# 0. run config_simple_sim_lab121_lidar.py to generate the .yaml configuration file
# 1. run in a terminal 'roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=lab121_lidar'
# 2. run in a terminal 'roslaunch sloop_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose'
# 3. run in a terminal 'roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=lab121_lidar'
# 4. run in a terminal 'python -m sloop_object_search.grpc.server'
# 5. run in a terminal 'python test_simple_sim_env_local_search.py'
# 6. run in a terminal 'roslaunch sloop_object_search_ros view_simple_sim.launch'
# ------------------

class TestSimpleEnvHierSearch:
    def __init__(self, o3dviz=False, prior="uniform"):
        # This is an example of how to get started with using the
        # sloop_object_search grpc-based package.
        rospy.init_node("test_simple_env_hier_search")


def main():
    TestSimpleEnvHierSearch(o3dviz=False, prior="uniform")

if __name__ == "__main__":
    main()
