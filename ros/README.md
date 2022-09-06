# Example ROS package that uses sloop_object_search

## Setup

### Create symbolic link

Go to the `src` folder of your ROS workspace. Then run:
```
ln -s path/to/sloop_object_search/ros sloop_object_search_ros
```
This effectively adds a ROS package called "sloop_object_search_ros" into your workspace

### Install Dependencies

This is a ROS package; Therefore, it is expected to operate within a ROS workspace.

Before building this package, make sure you have activated a virtualenv. Then, run
```
source install_dependencies.bash
```
to install python dependencies.

### Build the ROS package
```
catkin_make -DCATKIN_WHITELIST_PACKAGES="sloop_object_search_ros"
```



### Download Dataset and Models
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
