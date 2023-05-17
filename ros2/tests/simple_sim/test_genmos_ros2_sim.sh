#!/bin/bash

run_map_points_publisher="ros2 run genmos_object_search_ros2 simple_sim_map_points_publisher.py"
run_simple_sim_env="ros2 launch genmos_object_search_ros2 simple_sim_env_ros2.launch map_name:=lab121_lidar"
run_local_cloud_publisher="ros2 launch genmos_object_search_ros2 local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose global_cloud_topic:=graphnav_points"
run_genmos_server="python -m genmos_object_search.grpc.server"
run_genmos_searcher="ros2 launch genmos_object_search_ros2 simple_sim_search.launch map_name:=lab121_lidar"
run_rviz="ros2 run genmos_object_search_ros2 view_simple_sim_ros2.sh"

cat << EOF > /tmp/bootstrap_tabs.sh
gnome-terminal --tab -t "graphnav_map" -- \
  bash -c "$run_map_points_publisher; bash"
gnome-terminal --tab -t "simple_sim" -- \
  bash -c "sleep .75; $run_simple_sim_env; bash"
gnome-terminal --tab -t "local_cloud" -- \
  bash -c "sleep .75; $run_local_cloud_publisher; bash"
gnome-terminal --tab -t "genmos_server" -- \
  bash -c "sleep 2.0; $run_genmos_server; bash"
gnome-terminal --tab -t "genmos_client" -- \
  bash -c "sleep 3.0; $run_genmos_searcher; bash"
gnome-terminal --tab -t "rviz" -- \
  bash -c "sleep 1.0; $run_rviz; bash"
EOF

gnome-terminal --window -- bash /tmp/bootstrap_tabs.sh
