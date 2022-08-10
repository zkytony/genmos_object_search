# sloop_object_search

SLOOP (Spatial Language Understanding Object-Oriented POMDP)
for Multi-Object Search.


## Overview
This repository contains implementation of SLOOP for object search and related
planning algorithms which are **robot-middleware-independent**.  Referred to as
"sloop_object_search", this implementation allows:

1. Direct creation of SLOOP object search agents in Python through imports
2. Running the agent as a server that accepts gRPC calls.

The first option allows developers to build upon the source code of the POMDP agent.
The second option allows the agent to be run as a "backend" of a robotic system,
regardless of the middleware, as long as proper gRPC calls are made. These gRPC
calls serve the purpose of:

- belief update
- request planning next action
- execute an action
- information getters

The repository is structured such that the core middleware-independent code
is under "sloop_object_search", and each middleware that we have considered
has a corresponding folder.


## Set up sloop_object_search for ROS

1. Go to the 'src' folder of your ROS workspace.
2. Create a symbolic link to the `sloop_object_search/ros` folder, and name that
   symbolic link "sloop_object_search." That is,
   ```
   ln -s /path/to/sloop_object_search/ros sloop_object_search_ros
   ```
3. Compile the package
   ```
   catkin_make -DCATKLIN_WHITELIST_PACKAGES="sloop_object_search_ros"
   ```
