#!/bin/bash

run_graphnav_map_publisher="ros2 launch spot_funcs graphnav_map_publisher.launch map_name:=lab121_lidar"
run_simple_sim_env="ros2 launch genmos_object_search_ros2 simple_sim_env_ros2.launch map_name:=lab121_lidar"
run_local_cloud_publisher="ros2 launch genmos_object_search_ros2 local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose"
run_genmos_server="python -m genmos_object_search.grpc.server"
run_genmos_searcher="ros2 launch genmos_object_search_ros2 simple_sim_search.launch map_name:=lab121_lidar"
run_rviz="ros2 run genmos_object_search_ros2 view_simple_sim_ros2.sh"

cat << EOF > /tmp/bootstrap_tabs.sh
gnome-terminal --tab -t "graphnav_map" --working-directory="$BDAI" -- \
  bash -c "$run_graphnav_map_publisher; bash"
gnome-terminal --tab -t "simple_sim" --working-directory="$BDAI" -- \
  bash -c "sleep .75; $run_simple_sim_env; bash"
gnome-terminal --tab -t "local_cloud" --working-directory="$BDAI" -- \
  bash -c "sleep .75; $run_local_cloud_publisher; bash"
gnome-terminal --tab -t "genmos_server" --working-directory="$BDAI" -- \
  bash -c "sleep 2.0; $run_genmos_server; bash"
gnome-terminal --tab -t "genmos_client" --working-directory="$BDAI" -- \
  bash -c "sleep 3.0; $run_genmos_searcher; bash"
gnome-terminal --tab -t "rviz" --working-directory="$BDAI" -- \
  bash -c "sleep 1.0; $run_rviz; bash"
EOF

gnome-terminal --window -- bash /tmp/bootstrap_tabs.sh
