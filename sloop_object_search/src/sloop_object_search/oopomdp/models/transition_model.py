"""Defines the TransitionModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

Transition: deterministic
"""
import pomdp_py
from pomdp_py.framework.basics import State

import numpy as np

import copy
from ..domain.state import (ObjectState,
                            RobotState,
                            RobotStateTopo)

from ..domain.observation import *
from ..domain.action import *
from .sensors import yaw_facing, get_camera_direction3d, DEFAULT_3DCAMERA_LOOK_DIRECTION
from genmos_object_search.utils import math as math_utils


##################### Object Transition ##############################
class ObjectTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, objid):
        self.objid = objid

class StaticObjectTransitionModel(ObjectTransitionModel):
    """ static objects"""
    def probability(self, next_object_state, state, action):
        """
        Args:
            next_object_state (TargetObjectState): assumes to
                have the same object id as this transition model.
            state (OOState)
            action (pomdp_py.Action)
        Returns:
            float
        """
        if next_object_state == state.s(self.objid):
            return 1.0 - 1e-12
        else:
            return 1e-12

    def sample(self, state, action):
        """
        Args:
            state (OOState)
        Returns:
            ObjectState
        """
        return state.s(self.objid).copy()

##################### Robot Transition (Generic) ##############################
class RobotTransitionModel(ObjectTransitionModel):
    def __init__(self, robot_id, detection_models, no_look=False):
        super().__init__(robot_id)
        self.detection_models = detection_models
        self._no_look = no_look

    @property
    def robot_id(self):
        return self.objid

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        srobot = state.s(self.robot_id)
        current_robot_pose = srobot["pose"]
        next_robot_pose = current_robot_pose
        next_objects_found = srobot.objects_found
        next_camera_direction = srobot.camera_direction
        if self._no_look:
            next_camera_direction = action.name

        if isinstance(action, MotionAction):
            next_robot_pose = self.sample_motion(state, action)

        elif isinstance(action, LookAction):
            next_camera_direction = action.name

        elif isinstance(action, FindAction):
            next_objects_found = tuple(
                set(next_objects_found)
                | set(self.objects_in_range(current_robot_pose, state)))

        return RobotState(self.robot_id,
                          next_robot_pose,
                          next_objects_found,
                          next_camera_direction)

    def objects_in_range(self, robot_pose, state):
        objects_in_range = []
        for objid in state.object_states:
            if objid == self.robot_id:
                continue
            object_pose = state.s(objid)['pose']
            if self.detection_models[objid].sensor.in_range_facing(object_pose, robot_pose):
                objects_in_range.append(objid)
        return objects_in_range

    def sample_motion(self, state_or_pose, action):
        """Given a state or a pose, and a motion action,
        returns the resulting pose
        """
        if type(state_or_pose) != tuple:
            state = state_or_pose
            if isinstance(state, RobotState):
                srobot = state
            elif isinstance(state, pomdp_py.OOState):
                srobot = state.s(self.robot_id)
            else:
                raise ValueError(f"Invalid state type {type(state)}")
            return self.sample_by_pose(srobot.pose, action)
        else:
            pose = state_or_pose
            return self.sample_by_pose(pose, action)


##################### Robot Transition (2D) ##############################
class RobotTransBasic2D(RobotTransitionModel):
    def __init__(self, robot_id, detection_models,
                 reachability_func, no_look=False, **transform_kwargs):
        """Whether a position is reachable is determined by 'reachability_func'.
        This function could be implemented by the agent that uses this model,
        which has access to all the models and parameters that may determine
        reachability."""
        super().__init__(robot_id, detection_models, no_look=no_look)
        self.reachability_func = reachability_func
        self.transform_kwargs = transform_kwargs

    def sample_by_pose(self, pose, action):
        return RobotTransBasic2D.transform_pose(
            pose, action, self.reachability_func,
            **self.transform_kwargs)

    @classmethod
    def transform_pose(cls, pose, action,
                       reachability_func=None,
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
        rx = rx + forward*math.cos(math_utils.to_rad(rth))
        ry = ry + forward*math.sin(math_utils.to_rad(rth))
        rx, ry, rth = (*math_utils.fround(pos_precision, (rx, ry)),
                       math_utils.fround(rot_precision, rth))
        if reachability_func is not None:
            if reachability_func((rx, ry)):
                return (rx, ry, rth)
            else:
                return original_pose
        else:
            return (rx, ry, rth)


##################### Robot Transition (3D) ##############################
class RobotTransBasic3D(RobotTransitionModel):
    """robot movements over 3D grid"""
    def __init__(self, robot_id, reachability_func,
                 detection_models, no_look=False,
                 **transform_kwargs):
        """
        Note that the detection models are expected to contain 3D frustum
        camera models.

        transform_kwargs:
            pos_precision (default: 'int'): precision setting for transformed position
            rot_precision (default: '0.001'): precision setting for transformed rotation
            default_camera_direction (tuple): DEFAULT_3DCAMERA_LOOK_DIRECTION
                [used by forward motion scheme]. Note that this corresponds to
                the default robot's forward direction. If the robot has multiple
                cameras, this should be set to the robot's default forward direction.
                Note that the default camera direction is (0,0,-1) as the camera
                model (FrustumCamera) by default looks at -z.
        """
        super().__init__(robot_id, detection_models, no_look=no_look)
        self.reachability_func = reachability_func
        self.transform_kwargs = transform_kwargs

    def sample_by_pose(self, pose, action):
        return RobotTransBasic3D.transform_pose(
            pose, action, self.reachability_func,
            **self.transform_kwargs)

    @classmethod
    def transform_pose(cls, pose, action,
                       reachability_func=None,
                       **kwargs):
        """
        Args:
            pose (tuple): 7-element tuple that specify position and rotation.
            action (MotionAction3D or tuple): The underlying motion is a tuple.
                We can deal with two schemes of actions: 'axis' or 'forward',
                which are identified by whether the first element of this action tuple
                is a single number 'dforward' of a tuple (dx, dy, dz).
            reachability_func (function): takes in a position (x,y,z) and returns
                True if it is reachable as a position for the robot's viewpoint.
                If the transformed pose is not reachable, then the original pose will
                be returned, indicating no transition happened.
        """
        if type(action) == tuple:
            motion = action
        elif isinstance(action, MotionAction3D):
            motion = action.motion
        else:
            raise TypeError(f"action {action} is of invalid type")

        if type(motion[0]) == tuple:
            return cls._transform_pose_axis(pose, motion,
                                            reachability_func=reachability_func,
                                            **kwargs)
        else:
            return cls._transform_pose_forward(pose, motion,
                                               reachability_func=reachability_func,
                                               **kwargs)

    @classmethod
    def _transform_pose_axis(cls, pose, motion,
                             reachability_func=None,
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
        x, y, z = pose[:3]
        q = pose[3:]
        dpos, dth = motion

        # if len(dth) == 3:
        #     raise ValueError("Rotation motion should be specified by quaternion.")

        new_pos = math_utils.fround(pos_precision, (x+dpos[0], y+dpos[1], z+dpos[2]))
        if reachability_func is not None:
            if not reachability_func(new_pos):
                return pose

        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            q_change = math_utils.euler_to_quat(*dth)
            q_new = math_utils.quat_multiply(q_change, q)
            q = q_new
        return (*new_pos, *math_utils.fround(rot_precision, q))

    @classmethod
    def _transform_pose_forward(cls, pose, motion,
                                reachability_func=None,
                                pos_precision="int",
                                rot_precision=0.001,
                                default_camera_direction=DEFAULT_3DCAMERA_LOOK_DIRECTION):
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
            pose, default_camera_direction=default_camera_direction) # camere by default looks at (0,0,-1)
        forward, dth = motion

        # project this vector to xy plane, then obtain the "shadow" on xy plane
        forward_vec = robot_facing*forward
        xy_shadow = forward_vec - math_utils.proj(forward_vec, np.array([0,0,1]))
        dy = math_utils.proj(xy_shadow[:2], np.array([0,1]), scalar=True)
        dx = math_utils.proj(xy_shadow[:2], np.array([1,0]), scalar=True)
        yz_shadow = forward_vec - math_utils.proj(forward_vec, np.array([1,0,0]))
        dz = math_utils.proj(yz_shadow[1:], np.array([0,1]), scalar=True)

        dpos = (dx, dy, dz)
        x, y, z, qx, qy, qz, qw = pose
        new_pos = math_utils.fround(pos_precision, (x+dpos[0], y+dpos[1], z+dpos[2]))
        if reachability_func is not None:
            if not reachability_func(new_pos):
                return pose

        R = math_utils.R_quat(qx, qy, qz, qw)
        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = math_utils.R_quat(*math_utils.euler_to_quat(dth[0], dth[1], dth[2]))
            R = R_change * R_prev
        new_qrot = R.as_quat()
        return (*new_pos, *math_utils.fround(rot_precision, new_qrot))


##################### Robot Transition (Topo) ##############################
class RobotTransTopo(RobotTransitionModel):
    def __init__(self, robot_id, target_ids, topo_map,
                 detection_models, no_look=False,
                 **kwargs):
        """
        h_angle_res (float): resolution of horizontal rotation
            angle (in degrees) considered at the low level. It
            is used to create the set of rotation angles, which is used
            to sample a rotation angle facing the target as a
            result of a topo movement action.
        """
        super().__init__(robot_id, detection_models, no_look=no_look)
        self.topo_map = topo_map
        self._target_ids = target_ids

    def sample_motion(self, state, action):
        srobot = state.s(self.robot_id)
        if srobot.nid == action.src_nid:
            next_robot_pos = self.topo_map.nodes[action.dst_nid].pos
            next_topo_nid = action.dst_nid
            next_robot_rot = srobot.rot
            for target_id in self._target_ids:
                if target_id not in srobot.objects_found:
                    starget = state.s(target_id)
                    next_robot_rot = self.target_facing_rotation(
                        next_robot_pos, starget.loc)
                    break
            # If no more target to find, then just keep the current rotation
            if hasattr(next_robot_rot, '__len__'):
                next_robot_pose = (*next_robot_pos, *next_robot_rot)
            else:
                next_robot_pose = (*next_robot_pos, next_robot_rot)
            return (next_robot_pose, next_topo_nid)
        else:
            import traceback
            for line in traceback.format_stack():
                print(line.strip())
            print(":::::WARNING::::: Unexpected action {} for robot state {}. Ignoring action".format(action, srobot))
            return srobot['pose'], srobot['topo_nid']

    def target_facing_rotation(self, robot_pos, target_pos):
        """returns an orientation (domain-specific representation)
        which makes the robot face the target if the robot is
        at 'robot_pos' while the target is at 'target_pos'"""
        raise NotImplementedError()

    def argmax(self, state, action):
        srobot = state.s(self.robot_id)
        current_robot_pose = srobot["pose"]
        next_robot_pose = current_robot_pose
        next_objects_found = srobot.objects_found
        next_camera_direction = srobot.camera_direction
        next_topo_nid = srobot.topo_nid
        if self._no_look:
            next_camera_direction = action.name

        if isinstance(action, MotionAction):
            next_robot_pose, next_topo_nid = self.sample_motion(state, action)

        elif isinstance(action, LookAction):
            next_camera_direction = action.name

        elif isinstance(action, FindAction):
            next_objects_found = tuple(
                set(next_objects_found)
                | set(self.objects_in_range(current_robot_pose, state)))

        return RobotStateTopo(self.robot_id,
                              next_robot_pose,
                              next_objects_found,
                              next_camera_direction,
                              next_topo_nid,
                              srobot.topo_map_hashcode)


    def update(self, topo_map):
        self.topo_map = topo_map


class RobotTransTopo2D(RobotTransTopo):
    def __init__(self, robot_id, target_ids, topo_map,
                 detection_models, no_look=False,
                 **kwargs):
        super().__init__(robot_id, target_ids, topo_map,
                         detection_models, no_look=no_look)

    def target_facing_rotation(self, robot_pos, target_pos):
        """returns a yaw angle"""
        # will sample a yaw facing the target object
        yaw = yaw_facing(robot_pos, target_pos)
        return yaw


class RobotTransTopo3D(RobotTransTopo):
    def __init__(self, robot_id, target_ids, topo_map,
                 detection_models, no_look=False,
                 **kwargs):
        super().__init__(robot_id, target_ids, topo_map,
                         detection_models, no_look=no_look)
        # This default camera direction is the camera's look direction
        # when quaternion is 0,0,0,1 (i.e. no rotation).
        self.default_camera_direction = kwargs.get("default_camera_direction",
                                                   DEFAULT_3DCAMERA_LOOK_DIRECTION)

    def target_facing_rotation(self, robot_pos, target_pos, rot_precision=0.001):
        """Return a quaternion that represents the rotation of the default camera
        direction vector to the vector robot_pos, target_pos
        """
        if robot_pos == target_pos:
            # no rotation needed
            return np.array([0., 0., 0., 1.])

        q = math_utils.quat_between(
            self.default_camera_direction, math_utils.vec(robot_pos, target_pos))
        return q
