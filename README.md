# sloop_ros
ROS Package for SLOOP (Spatial Language Understanding Object-Oriented POMDP)


# Build

## As part of robotdev/spot
(Optional) To enable rtags indexing in emacs (for C++):
```
export SPOT_ADDITIONAL_BUILD_OPTIONS=-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```
Then, to build just sloop_ros,
```
build_spot -DCATKIN_WHITELIST_PACKAGES="sloop_ros"
```
