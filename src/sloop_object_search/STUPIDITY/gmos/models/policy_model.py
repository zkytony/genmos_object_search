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
from ..domain.action import Done
from ..utils.math import euclidean_dist

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

    def set_observation_model(self, observation_model, use_heuristic=True):
        # Classes that inherit this class can override this
        # function to create action prior
        self._observation_model = observation_model
