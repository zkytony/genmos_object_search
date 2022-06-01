"""Policy model for 2D Multi-Object Search domain. 
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
import math
from ..domain.action import *
from ..domain.observation import *
from .components.sensor import euclidean_dist

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_map=None, no_look=False):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self._grid_map = grid_map
        self._no_look = no_look

    @property
    def grid_map(self):
        return self._grid_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]
    
    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        if self._no_look:
            return self.get_all_actions_no_look(state=state, history=history)
        else:
            return self.get_all_actions_with_look(state=state, history=history)

    def get_all_actions_with_look(self, state=None, history=None):
        """note: find can only happen after look."""
        can_find = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = set({Find}) if can_find else set({})
        if state is None:
            return ALL_MOTION_ACTIONS | {Look} | find_action
        else:
            if self._grid_map is not None:
                valid_motions =\
                    self._grid_map.valid_motions(self.robot_id,
                                                 state.pose(self.robot_id),
                                                 ALL_MOTION_ACTIONS)
                return valid_motions | {Look} | find_action
            else:
                return ALL_MOTION_ACTIONS | {Look} | find_action

    def get_all_actions_no_look(self, state=None, history=None):
        """note: find can only happen after look."""
        find_action = set({Find})
        if state is None:
            return ALL_MOTION_ACTIONS | find_action
        else:
            if self._grid_map is not None:
                valid_motions =\
                    self._grid_map.valid_motions(self.robot_id,
                                                 state.pose(self.robot_id),
                                                 ALL_MOTION_ACTIONS)
                return valid_motions | find_action
            else:
                return ALL_MOTION_ACTIONS | find_action            

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]



# Preferred policy, action prior.    
class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior):
        self.action_prior = action_prior
        super().__init__(self.action_prior.robot_id,
                         self.action_prior.grid_map,
                         no_look=self.action_prior.no_look)
        self.action_prior.set_motion_actions(ALL_MOTION_ACTIONS)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    
class GreedyActionPriorXY(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme"""
    def __init__(self, robot_id, grid_map, num_visits_init, val_init,
                 no_look=False):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.all_motion_actions = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self.no_look = no_look

    def set_motion_actions(self, motion_actions):
        self.all_motion_actions = motion_actions
        
    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        #
        # Also do not prefer actions that makes the robot rotate in place back
        # and forth.
        if self.all_motion_actions is None:
            raise ValueError("Unable to get preferred actions because"\
                             "we don't know what motion actions there are.")
        robot_state = state.object_states[self.robot_id]

        last_action = None
        if len(history) > 0:
            last_action, last_observation = history[-1]
            for objid in last_observation.objposes:
                if objid not in robot_state["objects_found"]\
                   and last_observation.for_obj(objid).pose != ObjectObservation.NULL:
                    # We last observed an object that was not found. Then Find.
                    return set({(FindAction(), self.num_visits_init, self.val_init)})

        if self.no_look:
            # No Look action; It's embedded in Move.
            preferences = set()
        else:
            # Always give preference to Look
            preferences = set({(LookAction(), self.num_visits_init, self.val_init)})
        for objid in state.object_states:
            if objid != self.robot_id and objid not in robot_state.objects_found:
                object_pose = state.pose(objid)
                cur_dist = euclidean_dist(robot_state.pose, object_pose)
                neighbors =\
                    self.grid_map.get_neighbors(
                        robot_state.pose,
                        self.grid_map.valid_motions(self.robot_id,
                                                    robot_state.pose,
                                                    self.all_motion_actions))
                for next_robot_pose in neighbors:
                    if euclidean_dist(next_robot_pose, object_pose) < cur_dist:
                        action = neighbors[next_robot_pose]
                        preferences.add((action,
                                         self.num_visits_init, self.val_init))
        return preferences
    
class GreedyActionPriorVW(pomdp_py.ActionPrior):
    """greedy action prior for 'vw' motion scheme"""
    def __init__(self, robot_id, grid_map, num_visits_init, val_init,
                 no_look=False):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.all_motion_actions = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self.no_look = no_look

    def set_motion_actions(self, motion_actions):
        self.all_motion_actions = motion_actions
        
    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        # Prefer actions that move the robot closer to any
        # undetected target object in the state. If
        # cannot move any closer, look. If the last
        # observation contains an unobserved object, then Find.
        #
        # Also do not prefer actions that makes the robot rotate in place back
        # and forth.
        if self.all_motion_actions is None:
            raise ValueError("Unable to get preferred actions because"\
                             "we don't know what motion actions there are.")
        robot_state = state.object_states[self.robot_id]

        last_action = None
        if len(history) > 0:
            last_action, last_observation = history[-1]
            for objid in last_observation.objposes:
                if objid not in robot_state["objects_found"]\
                   and last_observation.for_obj(objid).pose != ObjectObservation.NULL:
                    # We last observed an object that was not found. Then Find.
                    return set({(FindAction(), self.num_visits_init, self.val_init)})

        if self.no_look:
            # No Look action; It's embedded in Move.
            preferences = set()
        else:
            # Always give preference to Look
            preferences = set({(LookAction(), self.num_visits_init, self.val_init)})
        for objid in state.object_states:
            if objid != self.robot_id and objid not in robot_state.objects_found:
                object_pose = state.pose(objid)
                cur_dist = euclidean_dist(robot_state.pose, object_pose)
                object_angle = (math.atan2(object_pose[1] - robot_state.pose[1],
                                           object_pose[0] - robot_state.pose[0])) % (2*math.pi)
                cur_angle_diff = abs(robot_state.pose[2] - object_angle)
                valid_motions = self.grid_map.valid_motions(self.robot_id,
                                                            robot_state.pose,
                                                            self.all_motion_actions)
                neighbors =\
                    self.grid_map.get_neighbors(
                        robot_state.pose,
                        valid_motions,
                        include_angle=True)
                for next_robot_pose in neighbors:
                    action = neighbors[next_robot_pose]
                    if euclidean_dist(next_robot_pose, object_pose) <= cur_dist:
                        # actually prefer forward as long as it doesn't move the robot away
                        preferences.add((action,
                                         self.num_visits_init, self.val_init))
                    else:
                        # also prefer rotation actions that brings the robot
                        # to the direction facing the target.
                        next_angle_diff = abs(next_robot_pose[2] - object_angle)
                        if next_angle_diff < cur_angle_diff:
                            preferences.add((action,
                                             self.num_visits_init, self.val_init))
        return preferences
                
    
    
