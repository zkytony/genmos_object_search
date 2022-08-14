# Copyright 2022 Kaiyu Zheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Coordinate conversion
#   gmapping origin is at lower-left. According to https://www.ros.org/reps/rep-0103.html,
# the X axis is East, Y axis is North.
#   Note that a point in the gmapping coordinate frame has unit in meter.
import math

class Frame:
    # gmapping map
    WORLD = "world"
    # a rectangular region in the gmapping map
    REGION = "region"
    # the POMDP search space.
    SEARCH_SPACE = "search_space"
    POMDP_SPACE = "search_space"        # .. alias
    POMDP_SEARCH_SPACE = "search_space" # .. alias

def convert(point, from_frame, to_frame,
            region_origin=None,
            search_space_resolution=None):
    """
    `from_frame` (str) one of {"world", "region", or "search_space"}
    `to_frame` (str) one of {"world", "region", or "search_space"}
    `region_origin` (tuple): coordinates of a point in the world frame
         that corresponds to (0,0) in the region, thus POMDP space
    `search_space_resolution` (float): the metric resolution of one
         coordinate length in the POMDP space (e.g. 0.15 means one
         POMDP grid equals to 0.15m in the real world)
    """
    all_frames = {Frame.WORLD, Frame.REGION, Frame.POMDP_SPACE}
    if not (from_frame.lower() in all_frames and to_frame.lower() in all_frames):
        raise ValueError("Unrecognized frame %s or %s" % (from_frame, to_frame))
    if from_frame == to_frame:
        return point

    if from_frame == Frame.WORLD:
        if to_frame == Frame.REGION:
            assert region_origin is not None,\
                "Converting from world to region requires origin of the region."
            return _world2region(point, region_origin)
        else:  # to_frame == search_space; convert twice
            region_point = convert(point, Frame.WORLD, Frame.REGION,
                                   region_origin=region_origin)
            return convert(region_point, Frame.REGION, Frame.POMDP_SPACE,
                           search_space_resolution=search_space_resolution)


    elif from_frame == Frame.REGION:
        if to_frame == Frame.POMDP_SPACE:
            assert search_space_resolution is not None,\
                "Converting from region to search_space requires resolution of the search space."
            return _region2searchspace(point, search_space_resolution)
        else: # to_frame == Frame.WORLD:
            assert region_origin is not None,\
                "Converting from region to world requires origin of the region."
            return _region2world(point, region_origin)

    elif from_frame == Frame.POMDP_SPACE:
        if to_frame == Frame.REGION:
            assert search_space_resolution is not None,\
                "Converting from search_space to region requires resolution of the search space."
            return _searchspace2region(point, search_space_resolution)
        else: # to_frame == Frame.WORLD; convert twice
            region_point = convert(point, Frame.SEARCH_SPACE, Frame.REGION,
                                   search_space_resolution=search_space_resolution)
            return convert(region_point, Frame.REGION, Frame.WORLD,
                           region_origin=region_origin)


def _world2region(world_point, region_origin):
    """
    Given `world_point` (x,y,z) in world coordinate (i.e. full gmapping map coordinate),
    and the origin of the rectangular region, also in world coordinate,
    returns the world point in region's coordinate frame. The region's
    points will have the same resolution as the gmapping map.
    """
    # simply subtract world point x,y by region origin. Keep z unchanged.
    region_point = tuple(world_point[i] - region_origin[i]
                         for i in range(len(region_origin)))
    if len(world_point) > len(region_origin):
        # we'd like to maintain the dimensionality
        region_point = (*region_point, *world_point[len(region_origin):])
    return region_point

def _region2world(region_point, region_origin):
    """region_point(x,y,z) -> world_point(x,y,z)"""
    world_point = tuple(region_point[i] + region_origin[i]
                        for i in range(len(region_origin)))
    if len(region_point) > len(region_origin):
        # we'd like to maintain the dimensionality
        world_point = (*world_point, *region_point[len(region_origin):])
    return world_point

def _region2searchspace(region_point, search_space_resolution):
    """Convert region point to a cube's coordinate in the search space.
    Assume that the search space's origin is at the region coordinate frame's
    origin. The `search_space_resolution` has unit m/cell."""
    return tuple(int(math.floor(region_point[i] / search_space_resolution))
                 for i in range(len(region_point)))

def _searchspace2region(search_space_point, search_space_resolution):
    return tuple(search_space_point[i] * search_space_resolution
                 for i in range(len(search_space_point)))
