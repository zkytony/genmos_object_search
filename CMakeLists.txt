cmake_minimum_required(VERSION 3.0.2)
project(sloop_ros)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  genmsg
  actionlib_msgs)

catkin_python_setup()


################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  DIRECTORY msg
  FILES
  KeyValAction.msg
  DefaultObservation.msg
  DefaultBelief.msg
  GridMapPose2d.msg
  GridMap2d.msg
  GridMapLoc.msg
  GridMapPose2d.msg
  Loc.msg
  SloopMosAction.msg
  SloopMosBelief.msg
  SloopMosJointObjectDetections.msg
  SloopMosObjectBelief.msg
  SloopMosObjectDetection.msg
  SloopMosSpatialLanguage.msg
  SloopMosRobotState.msg)

# Adding the action file for PlanAction
add_action_files(
  DIRECTORY action
  FILES PlanNextStep.action)

# generate_messages must be called AFTER catkin_python_setup
generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS actionlib_msgs)


############ Executables ##############
catkin_install_python(PROGRAMS scripts/run_agent
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

include_directories(
  ${catkin_INCLUDE_DIRS})
