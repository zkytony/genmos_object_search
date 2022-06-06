
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
from pomdp_py import RolloutPolicy, ActionPrior
from sloop_object_search.utils.math import euclidean_dist
from ..domain import action

class PolicyModel(RolloutPolicy):
    def __init__(self,
                 robot_trans_model,
                 num_visits_init=10,
                 val_init=100):
        self.robot_trans_model = robot_trans_model
        self.action_prior = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self._observation_model = None  # this can be helpful for the action prior

    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id

    @property
    def observation_model(self):
        return self._observation_model

    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state, history=None):
        raise NotImplementedError

    def rollout(self, state, history=None):
        if self.action_prior is not None:
            preferences = self.action_prior.get_preferred_actions(state, history)
            if len(preferences) > 0:
                return random.sample(preferences, 1)[0][0]
            else:
                return random.sample(self.get_all_actions(state=state), 1)[0]
        else:
            return random.sample(self.get_all_actions(state=state), 1)[0]


class PolicyModelBasic2D(PolicyModel):
    def __init__(self, robot_trans_model,
                 action_scheme,
                 observation_model,
                 no_look=False,
                 num_visits_init=10,
                 val_init=100):
        super().__init__(robot_trans_model, num_visits_init=10, val_init=100)
        self.movements = PolicyModelBasic2D.all_movements(action_scheme)
        self._no_look = no_look
        self._legal_moves = {}
        self._observation_model = observation_model
        if action_scheme == "vw":
            self.action_prior = PolicyModelBasic2D.ActionPriorVW(
                num_visits_init, val_init, self)

    @staticmethod
    def all_movements(action_scheme):
        if action_scheme == "vw":
            movements = {action.MoveForward,
                         action.MoveLeft,  # rotate left
                         action.MoveRight} # rotate right
        elif action_scheme == "xy":
            movements = {action.MoveNorth,
                         action.MoveSouth,
                         action.MoveEast,
                         action.MoveWest}
        else:
            raise ValueError(f"Invalid action scheme {action_scheme}")
        return movements

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
        return self.valid_moves(state) | {Look} | find_action

    def get_all_actions_no_look(self, state=None, history=None):
        """note: find can only happen after look."""
        return self.valid_moves(state) | {Find}

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
            valid_moves = set(a for a in self.movements
                if self.robot_trans_model.sample(state, a)["pose"] != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    ############# Action Prior VW ############
    class ActionPriorVW(ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            # If you have taken done before, you are done. So keep the done.
            last_action = history[-1][0] if len(history) > 0 else None
            if isinstance(last_action, Done):
                return {(Done(), 0, 0)}

            preferences = set()

            robot_id = self.policy_model.robot_id
            target_id = self.policy_model.observation_model.target_id
            srobot = state.s(robot_id)
            starget = state.s(target_id)
            if self.policy_model.reward_model.success(srobot, starget):
                preferences.add((Done(), self.num_visits_init, self.val_init))

            current_distance = euclidean_dist(srobot.loc, starget.loc)
            desired_yaw = yaw_facing(srobot.loc, starget.loc)
            current_yaw_diff = abs(desired_yaw - srobot.pose[2]) % 360

            for move in self.policy_model.movements:
                # A move is preferred if:
                # (1) it moves the robot closer to the target
                next_srobot = self.policy_model.robot_trans_model.sample(state, move)
                next_distance = euclidean_dist(next_srobot.loc, starget.loc)
                if next_distance < current_distance:
                    preferences.add((move, self.num_visits_init, self.val_init))
                    break

                # (2) if the move rotates the robot to be more facing the target,
                # unless the previous move was a rotation in an opposite direction;
                next_yaw_diff = abs(desired_yaw - next_srobot.pose[2]) % 360
                if next_yaw_diff < current_yaw_diff:
                    if hasattr(last_action, "dyaw") and last_action.dyaw * move.dyaw >= 0:
                        # last action and current are NOT rotations in different directions
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break

                # (3) it makes the robot observe any object;
                next_state = cospomdp.CosState({target_id: state.s(target_id),
                                                robot_id: next_srobot})
                observation = self.policy_model.observation_model.sample(next_state, move)
                for zi in observation:
                    if zi.loc is not None:
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break
            return preferences

    ############# Action Prior XY ############
    class ActionPriorXY(ActionPrior):
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
