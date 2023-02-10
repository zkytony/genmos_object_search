cmake_minimum_required(VERSION 3.0.2)
project(genmos_object_search_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  genmsg
  actionlib_msgs
  message_filters
  pcl_ros
  pcl_conversions)

catkin_python_setup()


################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  DIRECTORY msg
  FILES
  KeyValAction.msg
  KeyValObservation.msg)

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


############ Executables (Python) ##############
include_directories(
  ${catkin_INCLUDE_DIRS})


############ Executables (C++) ##############
# ~~~~~~~~~~ Setting up dependencies ~~~~~~ #
# Make Python headers available
find_package(PythonLibs ${PYTHON_MIN_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_MIN_VERSION 1.71)
find_package(Boost ${Boost_MIN_VERSION} REQUIRED)
set(Boost_LIBRARIES_USED
  boost_numpy
  boost_python)

# PCL
set(PCL_MIN_VERSION 1.10)
find_package(PCL ${PCL_MIN_VERSION} REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")

# roscpp and std_msgs and sensor_msgs
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})


################ Building our libraries and executables ###################
add_library(math_utils
  include/genmos_ros/utils/math_utils.cpp)
target_include_directories(math_utils
  PUBLIC
  include/genmos_ros/)

add_executable(local_cloud_publisher
  src/genmos_ros/processing/local_cloud_publisher.cpp)
target_include_directories(local_cloud_publisher
  PUBLIC
  include/genmos_ros/processing
  include/genmos_ros/)
target_link_libraries(local_cloud_publisher
  ${catkin_LIBRARIES})

add_executable(point_cloud_merger
  src/genmos_ros/processing/point_cloud_merger.cpp)
target_include_directories(point_cloud_merger
  PUBLIC
  include/genmos_ros/processing
  include/genmos_ros/)
target_link_libraries(point_cloud_merger
  ${catkin_LIBRARIES}
  yaml-cpp)

add_executable(occupancy_to_cloud
  src/genmos_ros/processing/occupancy_to_cloud.cpp)
target_include_directories(occupancy_to_cloud
  PUBLIC
  include/genmos_ros/processing
  include/genmos_ros/)
target_link_libraries(occupancy_to_cloud
  ${catkin_LIBRARIES})
