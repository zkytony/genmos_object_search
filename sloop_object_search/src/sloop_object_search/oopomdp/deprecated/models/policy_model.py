import math
import random
import pomdp_py
from genmos_object_search.utils.math import (euclidean_dist, fround)
from ...models.sensors import yaw_facing
from ...models.policy_model import PolicyModel
from ...domain import action
from ...domain.observation import ObjectDetection
from .transition_model import RobotTransTopo


######################### Policy Model Topo ########################
class PolicyModelTopo(PolicyModel):
    def __init__(self, target_ids,
                 robot_trans_model,
                 no_look=True,
                 num_visits_init=10,
                 val_init=100):
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
                        srobot.nid, nb_id, self.topo_map.edges[eid].nav_length))
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
