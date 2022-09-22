
ROS VERSION: ROS Kinetic

Note: PCL >= 1.11 required.

Note: Python 3 will be used when running the grpc client.
For that to connect properly with ROS kinetic, you need
to do ([ref](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)):
1. `sudo apt-get install python3-pip python3-yaml`
2. `sudo pip3 install rospkg catkin_pkg`
3. `sudo apt-get install python-catkin-tools python3-dev python3-numpy`
4. Install any dependency with `pip3 install ...` (note: not in a virtualenv)
   **OR:** Just run `pip3 install -e .` at `sloop_object_search`; This should install necessary dependencies
Note that virtualenv is not used at all.

Then, in the `run_search.py` node, the shebang should be
`#!/usr/bin/env python3`
