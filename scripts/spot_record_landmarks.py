#!/usr/bin/env python
# Record landmarks through object detection, and save them
# to the MapInfo database to allow spatial language
#
# Needs:
# - graphnav_map_publisher_with_localization.launch
#   (publishes map point cloud and robot localization)
#
# - stream_segmentation.py and publishing ROS messages
#
# - grid map
#
# Optional:
# - run the whole sloop_mos bridge with belief, if you'd like
#   to visualize the detection on a grid map.

#   Ideally, that the landmarks should be saved by their metric locations,
#   because the resolution of the grid map may change. But, I am coding this
#   now just for a demo.
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sloop_object_search.ros.ros_utils import tf2_transform
from rbd_spot_perception.msg import SimpleDetection3DArray
from sloop_ros.msg import GridMap2d
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_object_search.ros.mapinfo_utils import FILEPATHS, MapInfoDataset, register_map

def _confirm(message):
    while True:
        confirm = input("{} [y/n]: ".format(message))
        if confirm.lower().startswith('y'):
            return True
        elif confirm.lower().startswith('n'):
            return False

def add_landmark(mapinfo, map_name, landmark_symbol, landmark_footprint_grids):
    """
    adds a landmark with given footprint and symbol to the mapinfo dataset.
    """
    if landmark_symbol in mapinfo.landmarks[map_name]:
        print(f"WARNING: Landmark {landmark_symbol} will be overwritten.")
    mapinfo.landmarks[map_name][landmark_symbol] = list(sorted(landmark_footprint_grids))
    for loc in landmark_footprint_grids:
        if loc in mapinfo.cell_to_landmark(map_name):
            original_landmark_symbol = mapinfo.cell_to_landmark(map_name)[loc]
            print(f"WARNING: cell occupied originally by {original_landmark_symbol}. Now by {landmark_symbol}")
            mapinfo.cell_to_landmark(map_name)[loc] = landmark_symbol


class SpotLandmarkRecorder:
    def __init__(self,
                 map_name,
                 detection_3d_topic,   # published by stream_segmentation
                 grid_map_topic,   # published by stream_segmentation
                 map_frame="graphnav_map",
                 reject_overlaps=True,
                 confirm_landmarks=False):
        self.map_name = map_name

        self.grid_map = None

        # We would like to load the map and then modify its landmark information
        self.mapinfo = MapInfoDataset()
        self.mapinfo.load_by_name(self.map_name)

        self._cell_to_symbol = {}
        self._reject_overlaps = reject_overlaps
        self._confirm_landmarks = confirm_landmarks

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._map_frame = map_frame

        self._grid_map_sub = rospy.Subscriber("/graphnav_gridmap", GridMap2d, self._grid_map_cb)
        self._detection_3d_topic = detection_3d_topic
        self._det3d_sub = rospy.Subscriber(self._detection_3d_topic,
                                           SimpleDetection3DArray,
                                           self._detection_3d_cb)

    def grid_map_cb(self, grid_map_msg):
        if self.grid_map is None:
            self.grid_map = ros_msg_to_grid_map(grid_map_msg)

    def _detection_3d_cb(self, detection_msg):
        for det3d in detection_msg.detections:
            # Get detection position in map frame
            det_pose_stamped = PoseStamped(header=detection_msg.header, pose=det3d.box.center)
            det_pose_stamped_map_frame =\
                tf2_transform(self.tf_buffer, det_pose_stamped, self._map_frame)
            if det_pose_stamped_map_frame is None:
                # unable to get pose in map frame
                return

            # Get detected object pose in 2D. We will save the bounding
            # box as the footprint of the landmark
            obj_metric_position = det_pose_stamped_map_frame.position
            obj_metric_x, obj_metric_y = obj_metric_position.x, obj_metric_position.y
            landmark_metric_footprint = np.array([[obj_metric_x - det3d.box.size.x/2, obj_metric_y - det3d.box.size.y/2],
                                                  [obj_metric_x - det3d.box.size.x/2, obj_metric_y + det3d.box.size.y/2],
                                                  [obj_metric_x + det3d.box.size.x/2, obj_metric_y - det3d.box.size.y/2],
                                                  [obj_metric_x + det3d.box.size.x/2, obj_metric_y + det3d.box.size.y/2]])
            landmark_grid_topleft = self.grid_map.to_grid_pos(*landmark_metric_footprint[0])
            landmark_grid_bottomleft = self.grid_map.to_grid_pos(*landmark_metric_footprint[1])
            landmark_grid_topright = self.grid_map.to_grid_pos(*landmark_metric_footprint[2])
            landmark_grid_bottomright = self.grid_map.to_grid_pos(*landmark_metric_footprint[3])
            # obtain the grid cells within the box.
            landmark_footprint_grids = set()
            overlapping_symbols = {}
            for x in range(landmark_grid_topleft[0], landmark_grid_topright[0]+1):
                for y in range(landmark_grid_topleft[1], landmark_grid_bottomleft[1]+1):
                    if (x, y) in self._cell_to_symbol:
                        overlapping_symbols.add(self._cell_to_symbol[(x,y)])
                    landmark_footprint_grids.add((x, y))

            if self._reject_overlaps and len(overlapping_symbols) > 0:
                rospy.loginfo(f"Detected {det.label} but it is overlapping with {overlapping_symbols}")
                continue

            # Now, try to figure out a symbol for this landmark
            _count = 0
            landmark_symbol = "{}_{:03d}".format(det3d.label, _count)
            existing_landmarks = self.mapinfo.landmarks_for(self.map_name)
            while landmark_symbol in existing_landmarks:
                _count += 1
                landmark_symbol = "{}_{:03d}".format(det3d.label, _count)

            # ask whether to save this landmark
            if self._confirm_landmarks:
                if not _confirm(f"Save landmark {landmark_symbol}?"):
                    rospy.loginfo("landmark skipped.")
                    continue

            add_landmark(self.mapinfo, self.map_name, landmark_symbol, landmark_footprint_grids)
            rospy.loginfo("landmark added!")


    def done(self):
        # Ask for some synonyms
        name_to_symbols = {}
        symbol_to_synonyms = {}
        for landmark_symbol in self.mapinfo.landmarks[self.map_name]:
            names_input = input("Give some names to this symbol? (comma separated list): ")
            names = list(map(str.strip, names_input.split(",")))
            symbol_to_synonyms[landmark_symbol] = names
            for name in names:
                name_to_symbols[name] = landmark_symbol
        register_map(self.grid_map, exist_ok=True,
                     symbol_to_grids=self.mapinfo.landmarks[self.map_name],
                     name_to_symbols=name_to_symbols,
                     symbol_to_synonyms=symbol_to_synonyms)
