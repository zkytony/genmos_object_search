cmake_minimum_required(VERSION 3.0.2)
project(sloop_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs)

catkin_package()

catkin_python_setup()

add_executable(sloop_main src/sloop_ros/sloop_main)
target_link_libraries(sloop_main ${catkin_LIBRARIES})

include_directories(
  ${catkin_INCLUDE_DIRS})
