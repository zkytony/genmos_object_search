"""Defines the TransitionModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

Transition: deterministic
"""
import pomdp_py
from pomdp_py.framework.basics import State
import copy
from ..domain.state import (ObjectState,
                            ObjectState2D,
                            RobotState,
                            RobotState2D,
                            RobotStateTopo)

from ..domain.observation import *
from ..domain.action import *
from .sensors import yaw_facing
from sloop_object_search.utils.math import fround, to_rad

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

    def sample_motion(self, state, action, round_to="int"):
        """
        round_to (str): specifies rounding method of the location
            of transitioned pose. See utils.math.fround for definition.
        """
        raise NotImplementedError


class RobotTransBasic2D(RobotTransitionModel):
    """robot movements over 2D grid"""
    def __init__(self, robot_id, reachable_positions,
                 detection_models, **kwargs):
        super().__init__(robot_id, detection_models, **kwargs)
        self.reachable_positions = reachable_positions

    def sample_motion(self, state_or_pose, action, round_to="int"):
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

    def sample_by_pose(self, pose, action, round_to="int"):
        original_pose = pose
        rx, ry, rth = pose
        # odometry motion model
        forward, angle = action.motion
        rth = (rth + angle) % 360
        rx = rx + forward*math.cos(to_rad(rth))
        ry = ry + forward*math.sin(to_rad(rth))
        rx, ry, rth = fround(round_to, (rx, ry, rth))
        if (rx, ry) in self.reachable_positions:
            return (rx, ry, rth)
        else:
            return original_pose

    def argmax(self, state, action):
        srobot_next = super().argmax(state, action)
        return RobotState2D(srobot_next['id'],
                            srobot_next['pose'],
                            srobot_next['objects_found'],
                            srobot_next['camera_direction'])


class RobotTransTopo(RobotTransitionModel):
    def __init__(self, robot_id, target_ids, topo_map,
                 detection_models, h_angles, **kwargs):
        """
        h_angles (list): List of horizontal rotation (i.e. yaw)
            angles considered for planning
        """
        super().__init__(robot_id, detection_models, **kwargs)
        self.topo_map = topo_map
        self._target_ids = target_ids
        self._h_angles = h_angles

    def sample_motion(self, state, action, round_to="int"):
        srobot = state.s(self.robot_id)
        if srobot.nid == action.src_nid:
            next_robot_pos = self.topo_map.nodes[action.dst_nid].pos
            for target_id in self._target_ids:
                if target_id not in srobot.objects_found:
                    starget = state.s(target_id)
                    # will sample a yaw facing the target object
                    yaw = yaw_facing(next_robot_pos, starget.loc, self._h_angles)
                    next_robot_pose = (*next_robot_pos, yaw)
                    next_topo_nid = action.dst_nid
                    return (next_robot_pose, next_topo_nid)

            # If no more target to find, then just keep the current yaw
            next_robot_pose = (*next_robot_pos, srobot['pose'][2])
            return (next_robot_pose, next_topo_nid)
        else:
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
