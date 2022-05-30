cmake_minimum_required(VERSION 3.0.2)
project(sloop_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  genmsg
  actionlib_msgs)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

# Adding the action file for PlanAction
add_action_files(
  DIRECTORY action
  FILES PlanAction.action)

# generate_messages must be called AFTER catkin_python_setup
generate_messages(
  DEPENDENCIES actionlib_msgs)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS actionlib_msgs)


############ Executables ##############
catkin_install_python(PROGRAMS scripts/run_pomdp_agent
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
  ${catkin_INCLUDE_DIRS})
