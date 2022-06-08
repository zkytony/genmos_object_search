
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
import pomdp_py
from sloop_object_search.utils.math import (euclidean_dist)
from .sensors import yaw_facing
from ..domain import action
from ..domain.observation import ObjectDetection
from .transition_model import RobotTransTopo

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self,
                 robot_trans_model,
                 no_look=False,
                 num_visits_init=10,
                 val_init=100):
        self.robot_trans_model = robot_trans_model
        self.action_prior = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self.no_look = no_look
        self._observation_model = None  # this can be helpful for the action prior

    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id

    @property
    def observation_model(self):
        return self._observation_model

    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state=None, history=None):
        if self.no_look:
            return self.get_all_actions_no_look(state=state, history=history)
        else:
            return self.get_all_actions_with_look(state=state, history=history)

    def get_all_actions_with_look(self, state=None, history=None):
        """note: find can only happen after look."""
        can_find = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, action.LookAction):
                can_find = True
        find_action = set({action.Find}) if can_find else set({})
        return self.valid_moves(state) | {action.Look} | find_action

    def get_all_actions_no_look(self, state=None, history=None):
        """note: find can only happen after look."""
        return self.valid_moves(state) | {action.Find}

    def valid_moves(self, state):
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
    def __init__(self, target_ids,
                 robot_trans_model,
                 no_look=True,
                 num_visits_init=10,
                 val_init=100,
                 **action_args):
        super().__init__(robot_trans_model, no_look=no_look,
                         num_visits_init=10, val_init=100)
        self.movements = PolicyModelBasic2D.all_movements(**action_args)
        self.target_ids = target_ids
        self._legal_moves = {}
        self.action_prior = PolicyModelBasic2D.ActionPriorVW(
            num_visits_init, val_init, self)

    @staticmethod
    def all_movements(step_size=3, h_rotation=45.0):
        # scheme vw: (vt, vw) translational, rotational velocities.
        FORWARD = (step_size, 0)
        BACKWARD = (-step_size, 0)
        LEFT = (0, -h_rotation)  # left 45 deg
        RIGHT = (0, h_rotation)  # right 45 deg
        MoveForward = action.MotionAction2D(FORWARD, motion_name="Forward")
        MoveBackward = action.MotionAction2D(BACKWARD, motion_name="Backward")
        MoveLeft = action.MotionAction2D(LEFT, motion_name="TurnLeft")
        MoveRight = action.MotionAction2D(RIGHT, motion_name="TurnRight")
        movements = {"Forward": MoveForward,
                     "TurnLeft": MoveLeft,  # rotate left
                     "TurnRight": MoveRight} # rotate right
        return movements

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
            valid_moves = set(self.movements[a] for a in self.movements
                if self.robot_trans_model.sample(state, a)["pose"] != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    ############# Action Prior VW ############
    class ActionPriorVW(pomdp_py.ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            last_action = history[-1][0] if len(history) > 0 else None

            robot_id = self.policy_model.robot_id
            srobot = state.s(robot_id)
            if len(history) > 0:
                last_action, last_observation = history[-1]
                for objid in last_observation:
                    if objid != srobot["id"] and objid not in srobot["objects_found"]\
                       and last_observation.z(objid).pose != ObjectDetection.NULL:
                        # We last observed an object that was not found. Then Find.
                        return set({(action.FindAction(), self.num_visits_init, self.val_init)})

            if self.policy_model.no_look:
                preferences = set()
            else:
                preferences = set({(action.LookAction(), self.num_visits_init, self.val_init)})

            srobot = state.s(self.policy_model.robot_id)
            for move in self.valid_moves(state):
                srobot_next = self.policy_model.robot_trans_model.sample(state, move)
                for target_id in self.policy_model.target_ids:
                    starget = state.s(target_id)
                    # (1) 'move' brings the robot closer to target
                    if euclidean_dist(srobot_next.loc, starget.loc)\
                       < euclidean_dist(srobot.loc, starget.loc):
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break
                    # (2) After 'move', if robot moves foward, it gets closer to target
                    if move.motion[0] == 0.0:
                        srobot_nextnext_pose = self.policy_model.robot_trans_model.sample_motion(
                            srobot_next["pose"], self.policy_model.movements['Forward'])
                        if euclidean_dist(srobot_nextnext_pose[:2], starget.loc)\
                           < euclidean_dist(srobot.loc, starget.loc):
                            preferences.add((move, self.num_visits_init, self.val_init))
                            break
            return preferences


class PolicyModelTopo(PolicyModel):
    def __init__(self, target_ids,
                 robot_trans_model,
                 no_look=True,
                 num_visits_init=10,
                 val_init=100,
                 **action_args):
        assert no_look is True,\
            "planning over topological graph, no_look must be True!x"
        assert isinstance(robot_trans_model, RobotTransTopo),\
            "PolicyModelTopo needs RobotTransTopo as robot_trans_model"
        super().__init__(robot_trans_model, no_look=no_look,
                         num_visits_init=10, val_init=100)
        self.target_ids = target_ids
        self._legal_moves = {}
        self.action_prior = PolicyModelTopo.ActionPriorTopo(
            num_visits_init, val_init, self)

    @property
    def topo_map(self):
        return self.robot_trans_model.topo_map

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = srobot["pose"]
            valid_moves = {action.StayAction(srobot.nid)}  # stay is always a valid 'move'
            for nb_id in self.topo_map.neighbors(srobot.nid):
                eid = self.topo_map.edge_between(srobot.nid, nb_id)
                valid_moves.add(
                    action.MotionActionTopo(
                        srobot.nid, nb_id, self.topo_map.edges[eid].grid_dist))
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    def update(self, topo_map):
        """Update the topo_map"""
        self.robot_trans_model.update(topo_map)
        self._legal_moves = {}  # clear


    class ActionPriorTopo(pomdp_py.ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            last_action = history[-1][0] if len(history) > 0 else None

            robot_id = self.policy_model.robot_id
            srobot = state.s(robot_id)
            if len(history) > 0:
                last_action, last_observation = history[-1]
                for objid in last_observation:
                    if objid != srobot["id"] and objid not in srobot["objects_found"]\
                       and last_observation.z(objid).pose != ObjectDetection.NULL:
                        # We last observed an object that was not found. Then Find.
                        return set({(action.FindAction(), self.num_visits_init, self.val_init)})

            if self.policy_model.no_look:
                preferences = set()
            else:
                print("WARNING: planning over topological map; no_look should be True.")
                preferences = set({(action.LookAction(), self.num_visits_init, self.val_init)})

            srobot = state.s(robot_id)
            s_all_targets = {target_id: state.s(target_id)
                             for target_id in self.policy_model.target_ids}
            for move in self.policy_model.valid_moves(state):
                srobot_next = self.policy_model.robot_trans_model.sample(state, move)
                fake_next_state = pomdp_py.OOState({robot_id: srobot_next,
                                               **s_all_targets})
                for target_id in self.policy_model.target_ids:
                    starget = state.s(target_id)
                    # (1) 'move' brings the robot closer to target
                    if euclidean_dist(srobot_next.loc, starget.loc)\
                       < euclidean_dist(srobot.loc, starget.loc):
                        preferences.add((move, self.num_visits_init, self.val_init))
                        break

                    # (2) it is a stay, while any new object can be found
                    if isinstance(move, action.StayAction):
                        srobot_nextnext = self.policy_model.robot_trans_model.sample(
                            fake_next_state, move)
                        if len(srobot_nextnext.objects_found) > len(srobot.objects_found):
                            preferences.add((move, self.num_visits_init, self.val_init))
                            break
            return preferences
