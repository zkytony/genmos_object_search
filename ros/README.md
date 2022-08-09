# sloop_object_search
ROS Package for SLOOP (Spatial Language Understanding Object-Oriented POMDP)


## Overview
Currently structured as a ROS package, this repository contains implementation of
SLOOP for object search and related planning algorithms which are robot-middleware-independent.
Referred to as "sloop_object_search", this implementation allows:

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

If your middleware is not ROS, you can still clone this repository,
and install only the "sloop" and "sloop_object_search" packages, TODO


## Setup

### Install Dependencies

This is a ROS package; Therefore, it is expected to operate within a ROS workspace.

Before building this package, make sure you have activated a virtualenv. Then, run
```
source install_dependencies.bash
```
to install python dependencies.


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
