import math
from sloop_object_search.oopomdp.models.transition_model import RobotTransitionModel
from sloop_object_search.oopomdp.models.sensors import get_camera_direction3d, DEFAULT_3DCAMERA_LOOK_DIRECTION
from sloop_object_search.utils.math import (fround,
                                            to_rad,
                                            R_quat,
                                            euler_to_quat,
                                            proj)

##################### Robot Transition (2D) ##############################
class RobotTransBasic2D(RobotTransitionModel):
    """robot movements over 2D grid"""
    def __init__(self, robot_id, reachable_positions,
                 detection_models, no_look=False, **transform_kwargs):
        super().__init__(robot_id, detection_models, no_look=no_look)
        self.reachable_positions = reachable_positions
        self.transform_kwargs = transform_kwargs

    def sample_by_pose(self, pose, action):
        return RobotTransBasic2D.transform_pose(
            pose, action, self.reachable_positions,
            **self.transform_kwargs)

    @classmethod
    def transform_pose(cls, pose, action,
                       reachable_positions=None,
                       pos_precision="int",
                       rot_precision=0.001):
        original_pose = pose
        rx, ry, rth = pose
        # odometry motion model
        if type(action) == tuple:
            forward, angle = action
        else:
            forward, angle = action.motion
        rth = (rth + angle) % 360
        rx = rx + forward*math.cos(to_rad(rth))
        ry = ry + forward*math.sin(to_rad(rth))
        rx, ry, rth = (*fround(pos_precision, (rx, ry)),
                       fround(rot_precision, rth))
        if reachable_positions is not None:
            if (rx, ry) in reachable_positions:
                return (rx, ry, rth)
            else:
                return original_pose
        else:
            return (rx, ry, rth)



##################### Robot Transition (3D) ##############################
class RobotTransBasic3D(RobotTransitionModel):
    """robot movements over 3D grid"""
    def __init__(self, robot_id, reachable_positions,
                 detection_models, no_look=False,
                 **transform_kwargs):
        """
        Note that the detection models are expected to contain 3D frustum
        camera models.

        transform_kwargs:
            pos_precision (default: 'int'): precision setting for transformed position
            rot_precision (default: '0.001'): precision setting for transformed rotation
            default_forward_direction (tuple): DEFAULT_3DCAMERA_LOOK_DIRECTION
                [used by forward motion scheme]. Note that this corresponds to
                the default robot's forward direction. If the robot has multiple
                cameras, this should be set to the robot's default forward direction.
                Note that the default camera direction is (0,0,-1) as the camera
                model (FrustumCamera) by default looks at -z.
        """
        super().__init__(robot_id, detection_models, no_look=no_look)
        self.reachable_positions = reachable_positions
        self.transform_kwargs = transform_kwargs

    def sample_by_pose(self, pose, action):
        return RobotTransBasic3D.transform_pose(
            pose, action, self.reachable_positions,
            **self.transform_kwargs)

    @classmethod
    def transform_pose(cls, pose, action,
                       reachable_positions=None,
                       **kwargs):
        """
        Args:
            pose (tuple): 7-element tuple that specify position and rotation.
            action (MotionAction3D or tuple): The underlying motion is a tuple.
                We can deal with two schemes of actions: 'axis' or 'forward',
                which are identified by whether the first element of this action tuple
                is a single number 'dforward' of a tuple (dx, dy, dz).
            reachable_positions (set): set of positions (x, y, z) that are
                reachable. If the transformed pose is not reachable, then
                the original pose will be returned, indicating no transition happened.
        """
        if type(action) == tuple:
            motion = action
        elif isinstance(action, MotionAction3D):
            motion = action.motion
        else:
            raise TypeError(f"action {action} is of invalid type")

        if type(motion[0]) == tuple:
            return cls._transform_pose_axis(pose, motion,
                                            reachable_positions=reachable_positions,
                                            **kwargs)
        else:
            return cls._transform_pose_forward(pose, motion,
                                               reachable_positions=reachable_positions,
                                               **kwargs)

    @classmethod
    def _transform_pose_axis(cls, pose, motion,
                             reachable_positions=None,
                             pos_precision="int",
                             rot_precision=0.001,
                             **kwargs):
        """pose transform where the action is specified by change

        By default, motion specifies relative position and
        rotation change.

        Args:
            pos_precision ('int' or float): precision of position
            rot_precision ('int' or float): precision of rotation

        """
        x, y, z, qx, qy, qz, qw = pose
        R = R_quat(qx, qy, qz, qw)
        dpos, dth = motion

        # if len(dth) == 3:
        #     raise ValueError("Rotation motion should be specified by quaternion.")

        new_pos = fround(pos_precision, (x+dpos[0], y+dpos[1], z+dpos[2]))
        if reachable_positions is not None:
            if new_pos not in reachable_positions:
                return pose

        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = R_quat(*euler_to_quat(*dth))
            R = R_change * R_prev
        new_qrot = R.as_quat()
        return (*new_pos, *fround(rot_precision, new_qrot))

    @classmethod
    def _transform_pose_forward(cls, pose, motion,
                                reachable_positions=None,
                                pos_precision="int",
                                rot_precision=0.001,
                                default_forward_direction=DEFAULT_3DCAMERA_LOOK_DIRECTION):
        """
        pose transform where the action is specified by change

        Note that motion in action is specified by
           ((dx, dy, dz), (dthx, dthy, dthz))

        camera_default_look_direction (tuple): Used to calculate
            the current camera facing direction, which is given
            by pose, relative to this default look direction.
        """
        # We transform this direction vector to the given pose, which gives
        # us the current camera direction
        robot_facing = get_camera_direction3d(
            pose, default_camera_direction=default_forward_direction) # camere by default looks at (0,0,-1)
        forward, dth = motion

        # project this vector to xy plane, then obtain the "shadow" on xy plane
        forward_vec = robot_facing*forward
        xy_shadow = forward_vec - proj(forward_vec, np.array([0,0,1]))
        dy = proj(xy_shadow[:2], np.array([0,1]), scalar=True)
        dx = proj(xy_shadow[:2], np.array([1,0]), scalar=True)
        yz_shadow = forward_vec - proj(forward_vec, np.array([1,0,0]))
        dz = proj(yz_shadow[1:], np.array([0,1]), scalar=True)

        dpos = (dx, dy, dz)
        x, y, z, qx, qy, qz, qw = pose
        new_pos = fround(pos_precision, (x+dpos[0], y+dpos[1], z+dpos[2]))

        R = R_quat(qx, qy, qz, qw)
        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = R_quat(*euler_to_quat(dth[0], dth[1], dth[2]))
            R = R_change * R_prev
        new_qrot = R.as_quat()
        return (*new_pos, *fround(rot_precision, new_qrot))
