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
import numpy as np
import pomdp_py
from pomdp_py import RewardModel
from ..utils.math import euclidean_dist, to_rad
from ..domain.action import Find

class ObjectSearchRewardModel(pomdp_py.RewardModel):
    def __init__(self,
                 sensor, goal_dist, robot_id, target_id,
                 hi=100, lo=-100, step=-1):
        self.sensor = sensor
        self.goal_dist = goal_dist
        self.robot_id = robot_id
        self.target_id = target_id
        self._hi = hi
        self._lo = lo
        self._step = step  # default step cost

    def sample(self, state, action, next_state):
        srobot = state.s(self.robot_id)
        if srobot.done:
            return 0  # the robot is already done.
        starget = next_state.s(self.target_id)

        if isinstance(action, Find):
            if self.success(srobot, starget):
                # print("SUCCESS", srobot, starget)
                return self._hi
            else:
                return self._lo
        if hasattr(action, "step_cost"):
            return action.step_cost
        else:
            return self._step

    def success(self, srobot, starget):
        if euclidean_dist(srobot.loc, starget.loc) <= self.goal_dist:
            if srobot.in_range_facing(self.sensor, starget):
                return True
        return False
