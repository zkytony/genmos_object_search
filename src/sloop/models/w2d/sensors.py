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

import math
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

from sloop.utils.math import (to_rad, to_deg, R2d,
                              euclidean_dist, pol2cart,
                              in_range_inclusive, closest,
                              law_of_cos, inverse_law_of_cos)
from ..common import SensorModel

def yaw_facing(robot_pos, target_pos, angles=None):
    rx, ry = robot_pos
    tx, ty = target_pos
    yaw = to_deg(math.atan2(ty - ry,
                            tx - rx)) % 360
    if angles is not None:
        return closest(angles, yaw)
    else:
        return yaw

def pitch_facing(robot_pos3d, target_pos3d, angles=None):
    """
    Returns a pitch angle rotation such that
    if the robot is at `robot_position` and target is at
    `target_position`, the robot is facing the target.

    Args:
       robot_pos3d (tuple): x, y, z position
       target_pos3d (tuple): x, y, z position
       angles (list): Valid pitch angles (possible values for pitch
           in ai2thor agent rotation). Note that negative
           negative is up, positive is down
    Returns:
        .pitch angle between 0 - 360 degrees
    """
    rx, _, rz = robot_pos3d
    tx, _, tz = target_pos3d
    pitch = to_deg(math.atan2(tx - rx,
                              tz - rz)) % 360
    if angles is not None:
        return closest(angles, pitch)
    else:
        return pitch

# sensor_pose is synonymous to robot_pose, outside of this file.

class FanSensor(SensorModel):

    def init_params(self, name, **params):
        fov = params.get("fov", 90)
        min_range = params["min_range"]
        max_range = params["max_range"]
        self.name = name
        self.fov = fov  # convert to radian
        self._fov_rad = to_rad(fov)
        self.min_range = min_range
        self.max_range = max_range
        # this is not actually used unless the sensor model is far range.
        self.mean_range = params.get("mean_range", max_range)

    def __init__(self, name="laser2d_sensor", **params):
        """
        2D fanshape sensor. The sensor by default looks at the +x direction.
        The field of view angles span (-FOV/2, 0) U (0, FOV/2) degrees
        """
        self.init_params(name, **params)
        # The size of the sensing region here is the area covered by the fan
        # This is a float, but rounding it up should equal to the number of discrete locations
        # in the field of view.
        self._sensing_region_size = int(math.ceil(self._fov_rad / (2*math.pi) * math.pi * (self.max_range - self.min_range)**2))

    def __eq__(self, other):
        if not isinstance(other, FanSensor):
            return False
        if abs(self.fov - other.fov) <= 1e-4\
           and abs(self.min_range - other.min_range) <= 1e-4\
           and abs(self.max_range - other.max_range) <= 1e-4\
           and abs(self.mean_range - other.mean_range) <= 1e-4:
            return True
        else:
            return False

    def __str__(self):
        return f"FanSensor({self.min_range, self.max_range, self.mean_range, to_deg(self.fov)})"

    def __repr__(self):
        return str(self)

    def uniform_sample_sensor_region(self, sensor_pose):
        """Returns a location in the field of view
        uniformly at random. Expecting robot pose to
        have x, y, th, where th is in radians."""
        assert len(sensor_pose) == 3,\
            "Robot pose must have x, y, th"
        # Sample a location (r,th) for the default robot pose
        th = random.uniform(0, self._fov_rad) - self._fov_rad/2
        r = random.uniform(self.min_range, self.max_range)
        x, y = pol2cart(r, th)
        # transform to robot pose
        x, y = np.matmul(R2d(sensor_pose[2]), np.array([x,y])) # rotation
        x += sensor_pose[0]  # translation dx
        y += sensor_pose[1]  # translation dy
        point = (x, y)
        return point

    @property
    def sensor_region_size(self):
        return self._sensing_region_size

    def in_range(self, point, sensor_pose, use_mean=False):
        """
        Args:
            point (x, y): 2D point
            sensor_pose (x, y, th): 2D robot pose
        """
        if sensor_pose[:2] == point and self.min_range == 0:
            return True
        sensor_pose = (*sensor_pose[:2], to_rad(sensor_pose[2]))

        dist, bearing = self.shoot_beam(sensor_pose, point)
        range_bound = self.max_range if not use_mean else self.mean_range
        if self.min_range <= dist <= range_bound:
            # because we defined bearing to be within 0 to 360, the fov
            # angles should also be defined within the same range.
            fov_ranges = (0, self._fov_rad/2), (2*math.pi - self._fov_rad/2, 2*math.pi)
            if in_range_inclusive(bearing, fov_ranges[0])\
               or in_range_inclusive(bearing, fov_ranges[1]):
                return True
            else:
                return False
        return False

    def in_range_facing(self, point, sensor_pose,
                        angular_tolerance=15):
        desired_yaw = yaw_facing(sensor_pose[:2], point)
        current_yaw = sensor_pose[2]
        return self.in_range(point, sensor_pose)\
            and abs(desired_yaw - current_yaw) % 360 <= angular_tolerance

    def shoot_beam(self, sensor_pose, point):
        """Shoots a beam from sensor_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = sensor_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)
