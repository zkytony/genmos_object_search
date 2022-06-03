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
from ..domain.state import *
from ..domain.observation import *
from ..domain.action import *
from sloop_object_search.utils.math import fround


class RobotTransitionModel(pomdp_py.TransitionModel):
    """Models Pr(sr' | s, a); Likely domain-specific"""
    def __init__(self, robot_id):
        self.robot_id = robot_id

class ObjectTransitionModel(pomdp_py.TransitionModel):
    """Models Pr(si' | s, a); Likely domain-specific"""
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

def robot_pose_transition2d(robot_pose, action):
    """
    Uses the transform_pose function to compute the next pose,
    given a robot pose and an action.

    Note: robot_pose is a 2D POMDP (gridmap) pose.

    Args:
        robot_pose (x, y, th)
        action (Move2D)
    """
    rx, ry, rth = robot_pose
    forward, angle = action.delta
    nth = (rth + angle) % 360
    nx = rx + forward*math.cos(to_rad(nth))
    ny = ry + forward*math.sin(to_rad(nth))
    return (nx, ny, nth)


class RobotTransitionModel2D(RobotTransitionModel):
    def __init__(self, robot_id, reachable_positions, detection_models):
        """round_to: round the x, y coordinates to integer, floor integer,
        or not rounding, when computing the next robot pose."""
        super().__init__(robot_id)
        self.reachable_positions = reachable_positions
        self.detection_models = detection_models

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        srobot = state.s(self.robot_id)
        current_robot_pose = srobot["pose"]
        next_robot_pose = current_robot_pose
        next_found_objects = srobot.found_objects
        next_camera_direction = srobot.camera_direction

        if isinstance(action, MotionAction):
            np = robot_pose_transition2d(current_robot_pose, action)
            next_robot_pose = fround("int", np)
            if next_robot_pose[:2] not in self.reachable_positions:
                next_robot_pose = current_robot_pose

        elif isinstance(action, LookAction):
            next_camera_direction = action.name

        elif isinstance(action, FindAction):
            next_objects_found = tuple(set(next_objects_found)
                                       | set(objects_in_range(current_robot_pose, state)))

        return RobotState(self.robot_id,
                          next_robot_pose,
                          next_objects_found,
                          camera_direction)

    def objects_in_range(self, robot_pose, state):
        objects_in_range = []
        for objid in state.object_states:
            object_pose = state.s(objid)['pose']
            if self.detection_models[objid].sensor.in_range_facing(object_pose, robot_pose):
                objects_in_range.append(objid)
        return objects_in_range


class MosTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self, target_object_ids,
                 robot_transition_model):
        transition_models = {objid: StaticObjectTransitionModel(objid)
                             for objid in target_object_ids}
        transition_models[robot_id] = robot_transition_model
        super().__init__(transition_models)
