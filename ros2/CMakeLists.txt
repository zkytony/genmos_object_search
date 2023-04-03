cmake_minimum_required(VERSION 3.8)
project(genmos_object_search_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install genmos-related Python modules
ament_python_install_package(genmos_ros2)
install(
  PROGRAMS
  tests/simple_sim/simple_sim_env_ros2.py
  tests/simple_sim/simple_sim_init_pose_pub_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

# since ROS2 only accesses from within install,
install(
  DIRECTORY
  tests/
  DESTINATION share/${PROJECT_NAME}/tests/
)

# Messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyValAction.msg"
  "msg/KeyValObservation.msg"
  DEPENDENCIES builtin_interfaces
)

ament_package()
