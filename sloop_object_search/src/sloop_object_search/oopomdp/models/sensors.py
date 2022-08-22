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
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from sloop_object_search.utils.math import (to_rad, to_deg, R2d,
                                            euclidean_dist, pol2cart,
                                            vec, R_quat, R_euler, T, R_y,
                                            in_range_inclusive, closest,
                                            law_of_cos, inverse_law_of_cos,
                                            angle_between, in_box3d_origin)
from sloop_object_search.utils.algo import PriorityQueue
from sloop_object_search.oopomdp.models.octree_belief import OctNode

class SensorModel:
    IS_3D = False
    def in_range(self, point, sensor_pose):
        raise NotImplementedError

    def in_range_facing(self, point, sensor_pose,
                        angular_tolerance=15):
        """Returns True if the point is within the field of view,
        AND the sensor pose is facing the object directly,
        with some angular tolerance"""
        raise NotImplementedError


# sensor_pose is synonymous to robot_pose, outside of this file.

###################### 2D Fan Sensor ########################
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

## Utility functions
def yaw_facing(robot_pos, target_pos, angles=None):
    """robot_pos and target_pos are 2D"""
    rx, ry = robot_pos
    tx, ty = target_pos
    yaw = to_deg(math.atan2(ty - ry,
                            tx - rx)) % 360
    if angles is not None:
        return closest(angles, yaw)
    else:
        return yaw


###################### 3D Fan Sensor - tilted Fan Sensor ########################
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

## Utility functions
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


###################### 3D Frustum Sensor ########################
# By default, the camera is always at (0,0,0), looking at direction (0,0,-1).
# Direction of camera's look vector in camera's own frame
DEFAULT_3DCAMERA_LOOK_DIRECTION = (0, 0, -1)

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

    def __init__(self, fov=90, aspect_ratio=1, near=1,
                 far=5, occlusion_enabled=True):
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

        self._look = DEFAULT_3DCAMERA_LOOK_DIRECTION

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

    @property
    def look(self):
        return self._look

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
            self._look = get_camera_direction3d(
                pose, default_camera_direction=self.look)
        return p_moved, r_moved


    def in_range(self, point, sensor_pose):
        p, r = self.transform_camera(sensor_pose)
        x, y, z = point
        return self.within_range((p, r), (x, y, z, 1))

    def within_range(self, config, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion. To consider occlusion, see 'visible'"""
        p, r = config
        for i in range(6):
            if np.dot(vec(r[i], point), p[i]) >= 0:
                # print("Point outside plane %i" % i)
                # print("    Plane normal: %s" % str(p[i]))
                # print("    Plane refs: %s" % str(r[i]))
                # print("       Measure: %.3f" % np.dot(vec(r[i], point), p[i]))
                return False
        return True

    def in_range_facing(self, point, sensor_pose, tolerance=30):
        robot_facing = get_camera_direction3d(sensor_pose,
                                              default_camera_direction=self.look)
        robot_to_point = vec(sensor_pose[:3], point)
        angle_diff = angle_between(robot_facing, robot_to_point) % 360

        return self.in_range(point, sensor_pose)\
            and angle_diff <= tolerance

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
        point_in_norm = np.vmatmul(norm_mat, point_in_camera)

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

    def visible_volume(self, sensor_pose, occupancy_octree,
                       num_rays=20, step_size=0.1, return_obstacles_hit=False):
        """Return a voxelized volume that represents visible
        space if we put the camera at 'sensor_pose' and the
        environment contains occlusion due to occupancy_octree.
        Note that both sensor_pose and occupancy_octree should be in
        POMDP frame.

        If return_obstacles_hit is True, return a second set that
        is the set of obstacles hit by the rays"""
        # We shoot rays from the sensor out, and collect voxels
        # in the volume along the way, until the ray hits an obstacle.
        # The rays are sampled so that they hit a point on the near plane.

        # obstacles are at the leaf. Each obstacle is a voxel. We represent
        # them as origin-based boxes
        obstacles = [leaf for leaf in occupancy_octree.octree.get_leaves()
                     if leaf.value() > 0]
        # will use a priority queue, where higher priority means closer the distance to sensor pose
        obstacles_pq = PriorityQueue()
        for obstacle in obstacles:
            gx, gy, gz = obstacle.ground_origin
            # negative distance because priority queue favors smaller numbers
            obstacles_pq.push(OctNode.octnode_to_ground_box(obstacle),
                              -euclidean_dist(sensor_pose[:3], (gx, gy, gz)))

        visible_volume = set()
        obstacles_hit = set()
        w1, h1 = self._dim[:2]
        near = self._params[2]
        far = self._params[3]
        for i in tqdm(range(num_rays)):
            # sample a ray which goes through a point on the near plane
            ray_np_x = random.uniform(-w1/2, w1/2)
            ray_np_y = random.uniform(-h1/2, h1/2)
            ray_np_z = -self.near
            ray_up_angle = math.atan2(ray_np_y, ray_np_z)

            # transform the ray to (pomdp) world frame
            ray_np_world = self.camera_to_world((ray_np_x, ray_np_y, ray_np_z), sensor_pose)

            vec_ray = vec(sensor_pose[:3], ray_np_world)
            vec_ray = vec_ray / np.linalg.norm(vec_ray)  # normalize

            # advance on the ray, until it hits an obstacle
            t = 0
            hit_obstacle = False
            out_of_bound = False
            while not hit_obstacle and not out_of_bound:
                point_on_ray = sensor_pose[:3] + t * step_size * vec_ray

                if self._occlusion_enabled:
                    for obstacle_box in obstacles_pq:
                        if in_box3d_origin(point_on_ray, obstacle_box):
                            hit_obstacle = True
                            voxel = tuple(int(round(x)) for x in point_on_ray)  # ground-level voxel
                            # this obstacle is also visible
                            visible_volume.add(voxel)
                            obstacles_hit.add(voxel)
                            break

                if not hit_obstacle:
                    voxel = tuple(int(round(x)) for x in point_on_ray)  # ground-level voxel
                    visible_volume.add(voxel)

                t = t + 1

                # project the ray onto the z axis
                z_proj_ray = euclidean_dist(point_on_ray, sensor_pose[:3])*math.cos(ray_up_angle)
                if z_proj_ray < -far:
                    out_of_bound = True
        if return_obstacles_hit:
            return visible_volume, obstacles_hit
        else:
            return visible_volume

    def camera_to_world(self, point, sensor_pose):
        """Given a point in the camera frame, and a sensor_pose in
        the world frame, return the point in the world frame"""
        if len(sensor_pose) == 7:
            x, y, z, qx, qy, qz, qw = sensor_pose
            R = R_quat(qx, qy, qz, qw, affine=True)
        elif len(sensor_pose) == 6:
            x, y, z, thx, thy, thz = sensor_pose
            R = R_euler(thx, thy, thz, affine=True)
        # sensor_pose = T*R*(0,0,0);
        point_world = np.matmul(T(x,y,z), np.matmul(R, np.array([*point, 1])))
        return point_world[:3]


## Utility functions regarding 3D sensors
def get_camera_direction3d(current_pose,
                           default_camera_direction=DEFAULT_3DCAMERA_LOOK_DIRECTION):
    """
    Given a current 3D camera pose, return a
    vector that indicates its look direction.
    The default look direction is -z, or (0, 0, -1)

    Tip: default_camera_direction refers to the normal
    vector of the 0th plane in the frustum modeled by
    FrustumCamera.

    Args:
        current_pose (array-like)
    Returns:
        np.array
    """
    if len(current_pose) == 7:
        x, y, z, qx, qy, qz, qw = current_pose
        R = R_quat(qx, qy, qz, qw, affine=True)
    elif len(current_pose) == 6:
        x, y, z, thx, thy, thz = current_pose
        R = R_euler(thx, thy, thz, affine=True)
    d = np.array([*default_camera_direction, 1])
    d_transformed = np.transpose(np.matmul(R, np.transpose(d)))
    return d_transformed[:3]
