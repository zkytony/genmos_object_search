from sloop_ros.msg import GridMap2d
from sloop_object_search.ros.framework import ObservationInterpreter
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_object_search.ros.mapinfo_utils import FILEPATHS, register_map
from sloop_object_search.ros.sloop_mos import (grid_map_msg_callback,
                                               robot_pose_msg_callback)

class SpotObservationInterpreter(ObservationInterpreter):
    CALLBACKS = {"grid_map": grid_map_msg_callback,
                 "robot_pose": robot_pose_msg_callback}
