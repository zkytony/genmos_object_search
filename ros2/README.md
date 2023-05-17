# Example ROS2 package that uses genmos_object_search

This ROS2 package was developed under [ROS2 Humble](https://docs.ros.org/en/humble/index.html).

## Setup

### Create symbolic link

Go to the `src` folder of your ROS2 workspace. Then run:
```
ln -s path/to/genmos_object_search/ros2 genmos_object_search_ros2
```
This effectively adds a ROS2 package called “genmos_object_search_ros2” into your workspace.


### Install Dependencies
This is a ROS2 package; Therefore, it is expected to operate within a ROS2 workspace.

Before building this package, run
```
source install_dependencies.bash
```

### Build the ROS package
Go to the root of your ROS2 workspace directory, then do:
```
colcon_build --packages-select genmos_object_search_ros2
```
