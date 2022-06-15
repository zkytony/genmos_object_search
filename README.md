# sloop_ros
ROS Package for SLOOP (Spatial Language Understanding Object-Oriented POMDP)


# Setup

## Install Dependencies

This is a ROS package; Therefore, it is expected to operate within a ROS workspace.

Before building this package, make sure you have activated a virtualenv. Then, run
```
source install_dependencies.bash
```
to install python dependencies.


## Download Dataset and Models
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


Also, you will need to download spacy models. We use `en_core_web_lg` (400MB). To download it:
```
python -m spacy download en_core_web_lg
```


## Build the ROS package

Normally, you just need to run `catkin_make -DCATKIN_WHITELIST_PACKAGES="sloop_ros"`.

### As part of robotdev/spot
(Optional) To enable rtags indexing in emacs (for C++):
```
export SPOT_ADDITIONAL_BUILD_OPTIONS=-DCMAKE_EXPORT_COMPILE_COMMANDS=1
```
Then, to build just sloop\_ros,
```
build_spot -DCATKIN_WHITELIST_PACKAGES="sloop_ros"
```


## Test it out
Go to `sloop_ros/tests/sloop_object_search`, run any one (or all) of the following three tests:
```
python test_sloop_system.py config_test_basic2d.py
python test_sloop_system.py config_test_topo2d.py
python test_sloop_system.py config_test_hier2d.py
```
