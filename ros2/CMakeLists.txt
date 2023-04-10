cmake_minimum_required(VERSION 3.8)
project(genmos_object_search_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

############ Executables (Python) ##############
# ~~~~~~~~~~ Setting up dependencies ~~~~~~ #
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)

ament_python_install_package(genmos_ros2)
install(
  PROGRAMS
  tests/simple_sim/simple_sim_env_ros2.py
  tests/simple_sim/simple_sim_init_pose_pub_ros2.py
  scripts/run_search.py
  DESTINATION lib/${PROJECT_NAME}
)


############ Executables (C++) ##############
# ~~~~~~~~~~ Setting up dependencies ~~~~~~ #
find_package(rclcpp REQUIRED)

# PCL
set(PCL_MIN_VERSION 1.10)
find_package(PCL ${PCL_MIN_VERSION} REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# rclcpp and others
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
include_directories(${rclcpp_INCLUDE_DIRS})
include_directories(${rcl_interfaces_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})
include_directories(${geometry_msgs_INCLUDE_DIRS})
include_directories(${pcl_conversions_INCLUDE_DIRS})
include_directories(${yaml_cpp_vendor_INCLUDE_DIRS})

SET(Libraries
  pcl_conversions
  rclcpp
  rcl_interfaces
  std_msgs
  sensor_msgs
  geometry_msgs)

################ Building our libraries and executables ###################
add_library(math_utils
  include/genmos_ros2/utils/math_utils.cpp)
target_include_directories(math_utils
  PUBLIC
  include/genmos_ros2/)

add_executable(local_cloud_publisher
  src/genmos_ros2/processing/local_cloud_publisher.cpp)
target_include_directories(local_cloud_publisher
  PUBLIC
  include/genmos_ros2/processing
  include/genmos_ros2/)
ament_target_dependencies(local_cloud_publisher ${Libraries})
target_link_libraries(local_cloud_publisher ${PCL_LIBRARIES})

# add_executable(point_cloud_merger
#   src/genmos_ros2/processing/point_cloud_merger.cpp)
# target_include_directories(point_cloud_merger
#   PUBLIC
#   include/genmos_ros2/processing
#   include/genmos_ros2/)
# ament_target_dependencies(point_cloud_merger ${Libraries})
# target_link_libraries(point_cloud_merger ${YAML_CPP_LIBRARIES})


install(TARGETS
  local_cloud_publisher
  # point_cloud_merger
  DESTINATION lib/${PROJECT_NAME})


############ Files to install ##############
# since ROS2 only accesses from within install,
install(
  DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(
  DIRECTORY
  tests/
  DESTINATION share/${PROJECT_NAME}/tests
)

############ Messages ##############
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyValAction.msg"
  "msg/KeyValObservation.msg"
  DEPENDENCIES builtin_interfaces
)


ament_package()
