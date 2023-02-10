
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
from genmos_object_search.utils.math import (euclidean_dist, fround)
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

    @property
    def robot_id(self):
        return self.robot_trans_model.robot_id

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
                return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

######################### Policy Model Basic 2D ########################
class PolicyModelBasic2D(PolicyModel):
    def __init__(self, target_ids,
                 robot_trans_model,
                 primitive_movements,
                 no_look=True,
                 num_visits_init=10,
                 val_init=100):
        super().__init__(robot_trans_model, no_look=no_look,
                         num_visits_init=num_visits_init, val_init=val_init)
        self.movements = {a.motion_name: a for a in primitive_movements}
        self.target_ids = target_ids
        self._legal_moves = {}
        self.action_prior = PolicyModelBasic2D.ActionPrior(
            num_visits_init, val_init, self)

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = fround("int", srobot["pose"])
            valid_moves = set(self.movements[a] for a in self.movements
                if fround("int", self.robot_trans_model.sample(state, self.movements[a])["pose"]) != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    ############# Action Prior VW ############
    class ActionPrior(pomdp_py.ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            last_action = history[-1][0] if len(history) > 0 else None

            robot_id = self.policy_model.robot_id
            srobot = state.s(robot_id)

            if self.policy_model.no_look:
                preferences = {}
            else:
                preferences = {action.LookAction(): (self.num_visits_init, self.val_init)}

            srobot = state.s(self.policy_model.robot_id)
            for move in self.policy_model.valid_moves(state):
                srobot_next = self.policy_model.robot_trans_model.sample(state, move)
                for target_id in self.policy_model.target_ids:
                    if target_id in srobot.objects_found:
                        continue
                    starget = state.s(target_id)
                    # (1) 'move' brings the robot closer to target
                    if euclidean_dist(srobot_next.loc, starget.loc)\
                       <= euclidean_dist(srobot.loc, starget.loc):
                        preferences[move] = (self.num_visits_init, self.val_init)
                        break
            return {(a, preferences[a][0], preferences[a][1])
                    for a in preferences}

######################### Policy Model Topo ########################
class PolicyModelTopo(PolicyModel):
    def __init__(self, target_ids,
                 robot_trans_model,
                 no_look=True,
                 can_stay=True,
                 cost_scaling_factor=1.0,
                 num_visits_init=10,
                 val_init=100):
        assert no_look is True,\
            "planning over topological graph, no_look must be True!x"
        assert isinstance(robot_trans_model, RobotTransTopo),\
            "PolicyModelTopo needs RobotTransTopo as robot_trans_model"
        super().__init__(robot_trans_model, no_look=no_look,
                         num_visits_init=10, val_init=100)
        self.target_ids = target_ids
        self.can_stay = can_stay
        self._cost_scaling_factor = cost_scaling_factor
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
            valid_moves = set()
            if self.can_stay:
                valid_moves.add(action.StayAction(
                    srobot.nid, topo_map_hashcode=self.topo_map.hashcode))  # stay is always a valid 'move'
            for nb_id in self.topo_map.neighbors(srobot.nid):
                eid = self.topo_map.edge_between(srobot.nid, nb_id)
                valid_moves.add(
                    action.MotionActionTopo(
                        srobot.nid, nb_id, topo_map_hashcode=self.topo_map.hashcode,
                        cost_scaling_factor=self._cost_scaling_factor,
                        distance=self.topo_map.edges[eid].nav_length))
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
                       and last_observation.z(objid).label == objid:
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
                    if target_id in srobot.objects_found:
                        continue
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


######################### Policy Model Basic 3D ########################
class PolicyModelBasic3D(PolicyModel):
    def __init__(self, target_ids,
                 robot_trans_model,
                 primitive_movements,
                 no_look=True,
                 num_visits_init=10,
                 val_init=100):
        super().__init__(robot_trans_model, no_look=no_look,
                         num_visits_init=num_visits_init, val_init=val_init)
        self.movements = {a.motion_name: a for a in primitive_movements}
        self.target_ids = target_ids
        self._legal_moves = {}
        self.action_prior = PolicyModelBasic3D.ActionPrior(
            num_visits_init, val_init, self)

    def valid_moves(self, state):
        srobot = state.s(self.robot_id)
        if srobot in self._legal_moves:
            return self._legal_moves[srobot]
        else:
            robot_pose = fround("int", srobot["pose"])
            valid_moves = set(self.movements[a] for a in self.movements
                if fround("int", self.robot_trans_model.sample(state, self.movements[a])["pose"]) != robot_pose)
            self._legal_moves[srobot] = valid_moves
            return valid_moves

    ############# Action Prior VW ############
    class ActionPrior(pomdp_py.ActionPrior):
        def __init__(self, num_visits_init, val_init, policy_model):
            self.num_visits_init = num_visits_init
            self.val_init = val_init
            self.policy_model = policy_model

        def get_preferred_actions(self, state, history):
            last_action = history[-1][0] if len(history) > 0 else None

            robot_id = self.policy_model.robot_id
            srobot = state.s(robot_id)

            if self.policy_model.no_look:
                preferences = {}
            else:
                preferences = {action.LookAction(): (self.num_visits_init, self.val_init)}

            srobot = state.s(self.policy_model.robot_id)
            for move in self.policy_model.valid_moves(state):
                srobot_next = self.policy_model.robot_trans_model.sample(state, move)
                for target_id in self.policy_model.target_ids:
                    if target_id in srobot.objects_found:
                        continue
                    starget = state.s(target_id)
                    # (1) 'move' brings the robot closer to target
                    if euclidean_dist(srobot_next.loc, starget.loc)\
                       <= euclidean_dist(srobot.loc, starget.loc):
                        preferences[move] = (self.num_visits_init, self.val_init)
                        break
            return {(a, preferences[a][0], preferences[a][1])
                    for a in preferences}
