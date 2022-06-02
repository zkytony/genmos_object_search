cmake_minimum_required(VERSION 3.0.2)
project(sloop_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  genmsg
  actionlib_msgs
  pomdp_ros)

catkin_python_setup()

catkin_package()
