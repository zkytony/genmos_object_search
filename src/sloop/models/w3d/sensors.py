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
                              euclidean_dist,
                              vec, R_quat, R_euler, T, R_y,
                              in_range_inclusive, closest,
                              law_of_cos, inverse_law_of_cos)
from ..common import SensorModel
from ..w2d.sensors import FanSensor

class FrustumCamera(SensorModel):

    @property
    def near(self):
        return self._params[-2]

    @property
    def far(self):
        return self._params[-1]

    @property
    def fov(self):
        """returns fov in degrees"""
        return self._params[0] / math.pi * 180

    @property
    def aspect_ratio(self):
        return self._params[1]

    @property
    def volume(self):
        return self._volume

    def print_info(self):
        print("         FOV: " + str(self.fov))
        print("aspect_ratio: " + str(self.aspect_ratio))
        print("        near: " + str(self.near))
        print("         far: " + str(self.far))
        print(" volume size: " + str(len(self.volume)))

    def __init__(self, fov=90, aspect_ratio=1, near=1, far=5, occlusion_enabled=True):
        """
        fov: angle (degree), how wide the viewing angle is.
        near: near-plane's distance to the camera
        far: far-plane's distance to the camera
        """
        # Initially, the camera is always at (0,0,0), looking at direction (0,0,-1)
        # This can be changed by calling `transform_camera()`
        #
        # 6 planes:
        #     3
        #  0 2 4 5
        #     1

        # sizes of near and far planes
        fov = fov*math.pi / 180
        h1 = near * math.tan(fov/2) * 2
        w1 = abs(h1 * aspect_ratio)
        h2 = far * math.tan(fov/2) * 2
        w2 = abs(h2 * aspect_ratio)
        self._dim = (w1, h1, w2, h2)
        self._params = (fov, aspect_ratio, near, far)

        ref1 = np.array([w1/2, h1/2, -near, 1])
        ref2 = np.array([-w2/2, -h2/2, -far, 1])

        p1A = np.array([w1/2, h1/2, -near])
        p1B = np.array([-w1/2, h1/2, -near])
        p1C = np.array([w1/2, -h1/2, -near])
        n1 = np.cross(vec(p1A, p1B),
                      vec(p1A, p1C))

        p2A = p1A
        p2B = p1C
        p2C = np.array([w2/2, h2/2, -far])
        n2 = np.cross(vec(p2A, p2B),
                      vec(p2A, p2C))

        p3A = p1A
        p3B = p2C
        p3C = p1B
        n3 = np.cross(vec(p3A, p3B),
                      vec(p3A, p3C))

        p4A = np.array([-w2/2, -h2/2, -far])
        p4B = np.array([-w1/2, -h1/2, -near])
        p4C = np.array([-w2/2, h2/2, -far])
        n4 = np.cross(vec(p4A, p4B),
                      vec(p4A, p4C))

        p5A = p4B
        p5B = p4A
        p5C = p2B
        n5 = np.cross(vec(p5A, p5B),
                      vec(p5A, p5C))

        p6A = p4A
        p6B = p4C
        p6C = p2C
        n6 = np.cross(vec(p6A, p6B),
                      vec(p6A, p6C))

        # normal vectors for the six faces of the pyramid
        p = np.array([n1,n2,n3,n4,n5,n6])
        for i in range(6):  # normalize
            p[i] = p[i] / np.linalg.norm(p[i])
        p = np.array([p[i].tolist() + [0] for i in range(6)])
        r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])
        assert self.within_range((p, r), [0,0,-far-(-far+near)/2, 1])
        self._p = p
        self._r = r

        # compute the volume inside the frustum
        volume = []
        count = 0
        for z in range(-int(round(far)), -int(round(near))):
            for y in range(-int(round(h2/2))-1, int(round(h2/2))+1):
                for x in range(-int(round(w2/2))-1, int(round(w2/2))+1):
                    if self.within_range((self._p, self._r), (x,y,z,1)):
                        volume.append([x,y,z,1])
        self._volume = np.array(volume, dtype=int)
        self._occlusion_enabled = occlusion_enabled
        self._observation_cache = {}

    def transform_camera(self, pose, permanent=False):#x, y, z, thx, thy, thz, permanent=False):
        """Transformation relative to current pose; Affects where the sensor's field of view.
        thx, thy, thz are in degrees. Returns the configuration after the transform is applied.
        In other words, this is saying `set up the camera at the given pose`."""
        if len(pose) == 7:
            x, y, z, qx, qy, qz, qw = pose
            R = R_quat(qx, qy, qz, qw, affine=True)
        elif len(pose) == 6:
            x, y, z, thx, thy, thz = pose
            R = R_euler(thx, thy, thz, affine=True)
        r_moved = np.transpose(np.matmul(T(x, y, z),
                                         np.matmul(R, np.transpose(self._r))))
        p_moved =  np.transpose(np.matmul(R, np.transpose(self._p)))
        if permanent:
            self._p = p_moved
            self._r = r_moved
            self._volume = np.transpose(np.matmul(T(x, y, z),
                                                  np.matmul(R, np.transpose(self._volume))))
        return p_moved, r_moved


    def in_range(self, point, sensor_pose):
        p, r = self.transform_camera(sensor_pose)
        x, y, z = point
        return self.within_range((p, r), (x, y, z, 1))

    def within_range(self, config, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion"""
        p, r = config
        for i in range(6):
            if np.dot(vec(r[i], point), p[i]) >= 0:
                # print("Point outside plane %i" % i)
                # print("    Plane normal: %s" % str(p[i]))
                # print("    Plane refs: %s" % str(r[i]))
                # print("       Measure: %.3f" % np.dot(vec(r[i], point), p[i]))
                return False
        return True

    @property
    def config(self):
        return self._p, self._r

    # We need the notion of free space. The simplest thing to do
    # is for the sensor to directly inform the robot what the free
    # spaces are.
    def get_volume(self, sensor_pose, volume=None):
        """Return the volume inside the frustum as a list of 3D coordinates."""
        if volume is None:
            volume = self._volume
        if len(sensor_pose) == 7:
            x, y, z, qx, qy, qz, qw = sensor_pose
            R = R_quat(qx, qy, qz, qw, affine=True)
        elif len(sensor_pose) == 6:
            x, y, z, thx, thy, thz = sensor_pose
            R = R_euler(thx, thy, thz, affine=True)
        volume_moved = np.transpose(np.matmul(T(x, y, z),
                                              np.matmul(R, np.transpose(volume))))
        # Get x,y,z only
        volume_moved = volume_moved[:,:3]
        return np.round(volume_moved).astype(int)

    def field_of_view_size(self):
        return len(self._volume)

    def get_direction(self, p=None):
        if p is None:
            return -self._p[0][:3]
        else:
            return -p[0][:3]

    @staticmethod
    def sensor_functioning(alpha=1000., beta=0., log=False):
        """Utility used when sampling observation, to determine if the sensor works properly.
        (i.e. observed = True if the sensor works properly)

        log is true if we are dealing with log probabilities"""
        if log:
            # e^a / (e^a + e^b) = 1 / (e^{b-a} + 1)
            observed = random.uniform(0,1) < 1 / (math.exp(beta - alpha) + 1)
        else:
            observed = random.uniform(0,1) < alpha / (alpha + beta)
        return observed


    def perspectiveTransform(self, x, y, z, sensor_pose):
        # @params:
        # - x,y,z: points in world space
        # - sensor_pose: [eye_x, eye_y, eye_z, theta_x, theta_y, theta_z]

        point_in_world = [x,y,z,1.0]
        eye = sensor_pose[:3]
        rot = sensor_pose[3:]

        #default up and look vector when camera pose is (0,0,0)
        up = np.array([0.0, 1.0, 0.0])
        look = np.array([0.0, 0.0, -1.0])

        #transform up, look vector according to current camera pose
        r = R.from_quat([rot[0],rot[1],rot[2],rot[3]])
        curr_up = r.apply(up)
        curr_look = r.apply(look)
        curr_up += eye
        curr_look += eye

        #derive camera space axis u,v,w -> lookat Matrix
        w = - (curr_look - eye) / np.linalg.norm(curr_look - eye)
        v = curr_up - np.dot(curr_up, w) * w
        v = v / np.linalg.norm(v)
        u = np.cross(v,w)
        lookat = np.array([u, v, w])

        #Transform point in World Space to perspective Camera Space
        mat = np.eye(4)
        mat[0, 3] = -eye[0]
        mat[1, 3] = -eye[1]
        mat[2, 3] = -eye[2]
        point_in_camera = np.matmul(mat, point_in_world)

        axis_mat = np.eye(4)
        axis_mat[:3, :3] = lookat
        point_in_camera = np.matmul(axis_mat, point_in_camera)

        #Transform point in perspective Camera Space to normalized perspective Camera Space
        p_norm =  1.0 / ( self._params[3]  * np.tan(( self._params[0] * (np.pi/180.0) )/2) )
        norm_mat = np.eye(4, dtype = np.float32)
        norm_mat[0, 0] = p_norm
        norm_mat[1, 1] = p_norm
        norm_mat[2, 2] = 1.0 / self._params[-1]
        point_in_norm = np.matmul(norm_mat, point_in_camera)

        #Transform point in normalized perspective Camera Space to parallel camera viewing space
        c = - self._params[2] / self._params[3]
        unhinge_mat = np.eye(4, dtype=np.float32)
        unhinge_mat[2,2] = -1.0 / (1+c)
        unhinge_mat[2,3] = c / (1+c)
        unhinge_mat[3,2] = -1.0
        unhinge_mat[3,3] = 0.0
        point_in_parallel = np.matmul(unhinge_mat, point_in_norm)

        #De-homogenize
        point_in_parallel = point_in_parallel/ point_in_parallel[-1]

        return point_in_parallel



class FanSensor3D(FanSensor):
    """
    This is a simplified 3D sensor model; Instead of
    projecting a 3D volume, it re-shapes the 2D fan
    to account for the pitch of the camera pose. This
    sensor therefore can still be used to update 2D belief.

    All that is happening is the tilted Fan gets projected
    down to 2D.
    """
    IS_3D = True
    def __init__(self, name4="laser3d_sensor", **params):
        # Note that because of the tilt, the range will change.
        super().__init__(**params)
        self.v_angles = params["v_angles"]
        self._cache = {}

    @staticmethod
    def from_fan(fan, v_angles):
        return FanSensor3D(min_range=fan.min_range,
                           max_range=fan.max_range,
                           fov=fan.fov,
                           mean_range=fan.mean_range,
                           v_angles=v_angles)

    def __str__(self):
        return f"FanSensor3D({self.min_range, self.max_range, self.mean_range, to_deg(self.flat_fov)})"

    def __repr__(self):
        return str(self)

    def _project_range(self, height, pitch):
        mean_range_proj = self.mean_range * math.cos(to_rad(pitch))
        max_range_proj = self.max_range * math.cos(to_rad(pitch))
        min_range_proj = self.min_range * math.cos(to_rad(pitch))
        return min_range_proj, max_range_proj, mean_range_proj

    def _project_fov(self, pitch):
        # first we get the vector from fan center to the farthest points in the FOV.
        # Say the fan's origin is (0,0); the fan looks at +x by default;
        v1 = np.array([math.cos(to_rad(self.fov/2)),
                       math.sin(to_rad(self.fov/2)),
                       0,
                       1])
        v2 = np.array([math.cos(to_rad(self.fov/2)),
                       -math.sin(to_rad(self.fov/2)),
                       0,
                       1])  # height doesn't matter here

        # rotate both vectors with respect to the y axis by pitch
        v1_rot = np.dot(R_y(pitch), v1)
        v2_rot = np.dot(R_y(pitch), v2)

        # get the distance between the two vector end points on the x-y plane
        flat_fan_front_edge = euclidean_dist(v1_rot[:2], v2_rot[:2])
        fov_proj = inverse_law_of_cos(1., 1., flat_fan_front_edge)
        return fov_proj

    def _project2d(self, sensor_pose):
        if sensor_pose in self._cache:
            return self._cache[sensor_pose]
        else:
            x, y, height, pitch, yaw = sensor_pose
            params_proj = self._project_range(height, pitch)
            min_range_proj, max_range_proj, mean_range_proj = params_proj
            fov_proj = self._project_fov(pitch)
            fan2d = FanSensor(min_range=min_range_proj,
                              max_range=max_range_proj,
                              mean_range=mean_range_proj,
                              fov=fov_proj)
            self._cache[sensor_pose] = fan2d
        return fan2d

    def in_range(self, point, sensor_pose, use_mean=False):
        # Create a 2D sensor with projected parameters
        fan2d = self._project2d(sensor_pose)
        x, y, height, pitch, yaw = sensor_pose
        return fan2d.in_range(point, (x, y, yaw), use_mean=use_mean)

    def uniform_sample_sensor_region(self, sensor_pose):
        fan2d = self._project2d(sensor_pose)
        x, y, height, pitch, yaw = sensor_pose
        return fan2d.uniform_sample_sensor_region((x,y,yaw))

    def in_range_facing(self, point, sensor_pose,
                        angular_tolerance=15,
                        v_angular_tolerance=20):
        x, y, height, pitch, yaw = sensor_pose

        desired_yaw = yaw_facing(sensor_pose[:2], point[:2])
        current_yaw = sensor_pose[-1]

        desired_pitch = pitch_facing((x,y,height), point, self.v_angles)
        current_pitch = pitch

        return self.in_range(point, sensor_pose)\
            and abs(desired_yaw - current_yaw) % 360 <= angular_tolerance\
            and abs(desired_pitch - current_pitch) % 360 <= v_angular_tolerance
