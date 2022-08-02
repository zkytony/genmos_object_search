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
from .sensors import yaw_facing, get_camera_direction3d

from sloop_object_search.utils.math import (fround,
                                            to_rad,
                                            R_quat,
                                            euler_to_quat,
                                            proj)

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

        elif isinstance(action, ChangePoseAction):
            raise NotImplementedError("Transform by destination pose is not yet implemented")

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

    def sample_motion(self, state_or_pose, action, round_to="int"):
        """Given a state or a pose, and a motion action,
        returns the resulting pose

        round_to (str): specifies rounding method of the location
            of transitioned pose. See utils.math.fround for definition.

            TODO: round_to setting to int is not great for rotation.
                  should simply ask if the transition model is discrete
                  or continuo
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
    """robot movements over 2D grid"""
    def __init__(self, robot_id, reachable_positions,
                 detection_models, **kwargs):
        super().__init__(robot_id, detection_models, **kwargs)
        self.reachable_positions = reachable_positions

    def sample_by_pose(self, pose, action, round_to="int"):
        return RobotTransBasic2D.transform_pose(
            pose, action, self.reachable_positions, round_to=round_to)

    @classmethod
    def transform_pose(cls, pose, action,
                       reachable_positions=None, round_to="int"):
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
        rx, ry, rth = fround(round_to, (rx, ry, rth))
        if reachable_positions is not None:
            if (rx, ry) in reachable_positions:
                return (rx, ry, rth)
            else:
                return original_pose
        else:
            return (rx, ry, rth)


##################### Robot Transition (Topo) ##############################
class RobotTransTopo(RobotTransitionModel):
    def __init__(self, robot_id, target_ids, topo_map,
                 detection_models, h_angle_res=45.0, **kwargs):
        """
        h_angle_res (float): resolution of horizontal rotation
            angle (in degrees) considered at the low level. It
            is used to create the set of rotation angles, which is used
            to sample a rotation angle facing the target as a
            result of a topo movement action.
        """
        super().__init__(robot_id, detection_models, **kwargs)
        self.topo_map = topo_map
        self._target_ids = target_ids
        self._h_angles = [i*h_angle_res
                          for i in range(int(360/h_angle_res))]

    def sample_motion(self, state, action):
        srobot = state.s(self.robot_id)
        if srobot.nid == action.src_nid:
            next_robot_pos = self.topo_map.nodes[action.dst_nid].pos
            next_topo_nid = action.dst_nid
            for target_id in self._target_ids:
                if target_id not in srobot.objects_found:
                    starget = state.s(target_id)
                    # will sample a yaw facing the target object
                    yaw = yaw_facing(next_robot_pos, starget.loc, self._h_angles)
                    next_robot_pose = (*next_robot_pos, yaw)

                    return (next_robot_pose, next_topo_nid)

            # If no more target to find, then just keep the current yaw
            next_robot_pose = (*next_robot_pos, srobot['pose'][2])
            return (next_robot_pose, next_topo_nid)
        else:
            import traceback
            for line in traceback.format_stack():
                print(line.strip())
            print(":::::WARNING::::: Unexpected action {} for robot state {}. Ignoring action".format(action, srobot))
            return srobot['pose'], srobot['topo_nid']

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
                              next_topo_nid)


    def update(self, topo_map):
        self.topo_map = topo_map


##################### Robot Transition (3D) ##############################
class RobotTransBasic3D(RobotTransitionModel):
    """robot movements over 3D grid"""
    def __init__(self, robot_id, reachable_positions,
                 detection_models, **kwargs):
        """
        Note that the detection models are expected to contain 3D frustum
        camera models.
        """
        super().__init__(robot_id, detection_models, **kwargs)
        self.reachable_positions = reachable_positions

    def sample_by_pose(self, pose, action, **kwargs):
        return RobotTransBasic3D.transform_pose(
            pose, action, self.reachable_positions, **kwargs)

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
                             rot_precision=0.001):
        """
        pose transform where the action is specified by change

        Note that motion in action is specified by
           ((dx, dy, dz), (dthx, dthy, dthz))

        Args:
            pos_precision ('int' or float): precision of position
            rot_precision ('int' or float): precision of rotation
        """
        x, y, z, qx, qy, qz, qw = pose
        R = R_quat(qx, qy, qz, qw)
        dpos, dth = motion

        new_pos = fround(pos_precision, (x+dpos[0], y+dpos[1], z+dpos[2]))
        if reachable_positions is not None:
            if new_pos not in reachable_positions:
                return pose

        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = R_quat(*euler_to_quat(dth[0], dth[1], dth[2]))
            R = R_change * R_prev
        new_qrot = R.as_quat()
        return (*new_pos, *fround(rot_precision, new_qrot))

    @classmethod
    def _transform_pose_forward(cls, pose, motion,
                                reachable_positions=None,
                                pos_precision="int",
                                rot_precision=0.001):
        """
        pose transform where the action is specified by change

        Note that motion in action is specified by
           ((dx, dy, dz), (dthx, dthy, dthz))
        """
        # We transform this direction vector to the given pose, which gives
        # us the current camera direction
        robot_facing = get_camera_direction3d(pose) # camere by default looks at (0,0,-1)
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
