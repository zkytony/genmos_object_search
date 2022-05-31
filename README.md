# sloop_ros
ROS Package for SLOOP (Spatial Language Understanding Object-Oriented POMDP)


# Install Dependencies

This is a ROS package; Therefore, it is expected to operate within a ROS workspace.

Before building this package, make sure you have activated a virtualenv. Then, run
```
source install_dependencies.bash
```
to install python dependencies.


# Download Dataset and Models
Install gdown, a Python package used to download files from Google Drive:
```
pip install gdown
```
Then, run
```
python download.py
```
This will download both the SL\_OSM\_Dataset and the frame of reference prediction models.
The SL\_OSM\_Dataset will be saved under `data`, while the models
will be saved under `models`.


# Build the ROS package

Normally, you just need to run `catkin_make -DCATKIN_WHITELIST_PACKAGES="sloop_ros"`.

## As part of robotdev/spot
(Optional) To enable rtags indexing in emacs (for C++):
```
export SPOT_ADDITIONAL_BUILD_OPTIONS=-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```
Then, to build just sloop\_ros,
```
build_spot -DCATKIN_WHITELIST_PACKAGES="sloop_ros"
```
