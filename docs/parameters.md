Below are tables all parameters that is considered in the GenMOS system.

Each role of the table corresponds to an entry in the configuration dictionary.
The string in "Category" corresponds to the (sequence of) keys leading up
to the actual parameter, and the string in "Name of Parameter" corresponds to,
of course, the parameter name. For example, if "Category" is `a.b` and
"Name of Parameter" is `c`, then this corresponds to a dictionary structure
as follows:
```
"a": {
   "b": {
      "c": <parameter_value>
   }
}
```



To understand the information in this table:
* In `Category`, a string `a.b`



Parameters general across all middleware
| **Category**                  | **Name of Parameter** | **Description**                                                                                                                                                          | **Example**                                                             |
|-------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| agent_config                  | agent_class           | (str) Python class name of the GenMOS POMDP agent.                                                                                                                       | MosAgentBasic2D, MosAgentBasic3D, MosAgentTopo2D, MosAgentTopo3D        |
| agent_config                  | agent_type            | (str) Type of agent (in the planning hierarchy). Intended for differentiating an agent that focuses on local search and an agent that can deal with hierarchical search. | Must be 'local', 'hierarchical', or 'local_hierarchical'                |
| agent_config.search_region.3d | res                   | (float) octree resolution (along one dimension) in meters                                                                                                                | 0.1 means each cube in the octree covers 0.1x0.1x0.1 m^3 volume         |
| agent_config.search_region.3d | octree_size           | (int) Octree size (along one dimension) in number of ground-level nodes                                                                                                  | 32 means at the ground level, along each dimension, there are 32 cubes. |
| agent_config.search_region.3d | region_size_{x\|y\|z} | (float) length (in meters) of a box that encapsulates the search region.                                                                                                 | 3.0                                                                     |
| agent_config.search_region.3d | center_{x\|\y\|z}     | (float) world-frame coordinates (in meters) of the search region's center __position__.                                                                                  | -0.5                                                                    |
| agent_config.search_region.3d | center_q{x\|\y\|z}    | (float) world-frame coordinates (in meters) of the search region's center __orientation__.                                                                               | 0.78737748                                                              |
| agent_config.search_region.3d | debug                 | (bool) Turn on debug mode for search region initialization (an Open3D visualization window may pop up)                                                                    | False                                                                 |
|                               |                       |                                                                                                                                                                          |                                                                         |


Parameters specific to ROS2
| **Category** | **Name of Parameter** | **Description**                                                                                                          | **Example** |
|--------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------|-------------|
| ros2         | detection_class_names | (list) a list of names that should correspond one-to-one to the object detector model's list of possible output classes. | ["person", "bicycle", "car"...] Example from [YOLOv5 trained on COCO](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml)           |
