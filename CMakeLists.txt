cmake_minimum_required(VERSION 3.0.2)
project(sloop_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  genmsg
  actionlib_msgs)

add_action_files(
  DIRECTORY action
  FILES PlanAction.action)

generate_messages(
  DEPENDENCIES actionlib_msgs)

catkin_package()

catkin_python_setup()

add_executable(sloop_main src/sloop_ros/sloop_main)
target_link_libraries(sloop_main ${catkin_LIBRARIES})

add_executable(sloop_belief src/sloop_ros/sloop_belief)
target_link_libraries(sloop_belief ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS scripts/sloop_planning
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
  ${catkin_INCLUDE_DIRS})
