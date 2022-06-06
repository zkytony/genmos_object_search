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
                            RobotState2D)

from ..domain.observation import *
from ..domain.action import *
from sloop_object_search.utils.math import fround

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
    def __init__(self, robot_id, reachable_positions, detection_models):
        super().__init__(robot_id)
        self.reachable_positions = reachable_positions
        self.detection_models = detection_models

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

        if isinstance(action, MotionAction):
            next_robot_pose = self.motion_transition(srobot, action)

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

    def motion_transition(self, srobot, action, round_to="int"):
        """
        round_to (str): specifies rounding method of the location
            of transitioned pose. See utils.math.fround for definition.
        """
        raise NotImplementedError


class RobotTransBasic2D(RobotTransitionModel):
    """robot movements over 2D grid"""
    def __init__(self, robot_id, reachable_positions, detection_models, action_scheme):
        super().__init__(robot_id, reachable_positions, detection_models)
        self.action_scheme = action_scheme

    def motion_transition(self, srobot, action, round_to="int"):
        rx, ry, rth = srobot.pose

        if self.action_scheme == "xy":
            dx, dy, th = action.motion
            rx += dx
            ry += dy
            rth = th
        elif self.action_scheme == "vw":
            # odometry motion model
            forward, angle = action.motion
            rth += angle  # angle (radian)
            rx = rx + forward*math.cos(rth)
            ry = ry + forward*math.sin(rth)
            rth = rth % (2*math.pi)
        rx, ry = fround(round_to, (rx, ry))
        if (rx, ry) in self.reachable_positions:
            return (rx, ry, rth)
        else:
            return srobot['pose']

    def argmax(self, state, action):
        srobot_next = super().argmax(state, action)
        return RobotState2D(srobot_next['id'],
                            srobot_next['pose'],
                            srobot_next['objects_found'],
                            srobot_next['camera_direction'])
