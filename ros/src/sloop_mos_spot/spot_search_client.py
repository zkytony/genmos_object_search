# import numpy as np
# import time
# import rospy
# import pickle
# import json
# import sensor_msgs.msg as sensor_msgs
# import geometry_msgs.msg as geometry_msgs
# import std_msgs.msg as std_msgs
# from visualization_msgs.msg import Marker, MarkerArray
# from sloop_object_search_ros.msg import KeyValAction, KeyValObservation
# from sloop_object_search.grpc.client import SloopObjectSearchClient
# from sloop_object_search.grpc.utils import proto_utils
# from sloop_mos_ros import ros_utils
# from sloop_object_search.utils.open3d_utils import draw_octree_dist
# from sloop_object_search.grpc import sloop_object_search_pb2 as slpb2
# from sloop_object_search.grpc import observation_pb2 as o_pb2
# from sloop_object_search.grpc import action_pb2 as a_pb2
# from sloop_object_search.grpc import common_pb2
# from sloop_object_search.grpc.common_pb2 import Status
# from sloop_object_search.utils.colors import lighter
# from sloop_object_search.utils import math as math_utils

# class SpotSearchClient:
