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


# Documentation

The sloop_ros package can be thought of as wrapping a ROS interface around
a SLOOP agent, which allows the SLOOP agent to interact with real robot sensor
inputs and execute actions in the real world.

A SLOOP agent is a pomdp_py.Agent that represents the SLOOP POMDP. It refers
to a _class of POMDPs_ that have a spatial language observation model and
that factors state and observation spaces by objects.

## SLOOP Agent OUTDATED DOCS

There are currently two specific implementations of the SLOOP agent:
- **SloopMosBasic2DAgent** (at [basic2d.py](./src/sloop_object_search/oopomdp/agent/basic2d.py))

  Here, "Basic2D" means the action space of the agent is "basic": it consists
  of primitive motions such as moving forward, turning left and turning right
  (see [PolicyModelBasic2D](./src/sloop_object_search/oopomdp/models/policy_model.py)).

- **SloopMosTopo2DAgent** (at [topo2d.py](./src/sloop_object_search/oopomdp/agent/topo2d.py))

  Here, "Topo2D" means the action space of the agent is "topological": it
  consists of navigations between topological graph nodes.
  (see [PolicyModelTopo](./src/sloop_object_search/oopomdp/models/policy_model.py)).

A SLOOP agent can be most conveniently created based on a configuration dictionary.
See [tests/sloop_object_search/config_test_topo2d.py](./tests/sloop_object_search/config_test_topo2d.py)
as an example configuration dictionary. See [tests/sloop_object_search/test_sloop_system.py](./tests/sloop_object_search/test_sloop_system.py)
for how to actually create the agent. It boils down to:
```python
map_name = _config['task_config']["map_name"]
agent = eval(_config["agent_config"]["agent_class"])(
    _config["agent_config"], map_name)
```
Note that `_config` is the configuration dictionary.

Now, after creating the SLOOP agent, you'd like it to _act_. To do so, you need to
specify the configuration of a planner in the configuration dictionary. For example,
if you'd like to plan directly over the primitive actions (of the SloopMosBasic2DAgent),
the planner configuration could be:
```python
config = {
    "planner_config": {
        "planner": "pomdp_py.POUCT",
        "planner_params": {
            "max_depth": 20,
            "exploration_const": 1000,
            "planning_time": 0.25
        }
    }, ...
```
In fact, you can use the same planner configuration to plan for SloopMosTopo2DAgent;
This will produce navigation actions to topological nodes - the agent jumps around
in the visualization (test this out by running the test
`python test_sloop_system.py config_test_topo2d.py`)

There is a more sophisticated hierarchical planner you can specify. This planner
is meant for continuously running the SLOOP system, by carrying out each individual
primitive action to fulfill a navigation subgoal, or a stay subgoal in the action
space of a SloopMosTopo2DAgent. To use this planner:
```python
config = {
    "planner_config": {
        "planner": "sloop_object_search.oopomdp.planner.hier2d.HierarchicalPlanner",
        "subgoal_level": {
            "max_depth": 20,
            "exploration_const": 1000,
            "planning_time": 0.25
        },
        "local_search": {
            "planner": "pomdp_py.POUCT",
            "planner_args": {
                "max_depth": 10,
                "exploration_const": 1000,
                "planning_time": 0.15
            }
        }
    }, ...
```

## ROS Wrapper for SLOOP Agent

TODO


### How to Deploy SLOOP ROS on a Robot
TODO
