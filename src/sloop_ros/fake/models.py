import pomdp_py
import random
from sloop_ros.core.base_belief import BaseBelief
from sloop_ros.core.base_action import BaseAction

class FakeAction(BaseAction):
    pass


class FakeTransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if next_state == state:
            return 1.0 - 1e-9
        else:
            return 1e-9

    def sample(self, state, action):
        return pomdp_py.SimpleState(state.data)


class FakeObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action):
        if observation.data == next_state.data:
            return 1.0 - 1e-9
        else:
            return 1e-9

    def sample(self, next_state, action):
        return pomdp_py.SimpleObservation(next_state.data)


class FakeRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        if action.name == state.data:
            return 10
        else:
            return -5

class FakePolicyModel(pomdp_py.RolloutPolicy):
    ACTIONS = [FakeAction("left"),
               FakeAction("right")]
    def sample(self, state):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state=None, history=None):
        return FakePolicyModel.ACTIONS

    def rollout(self, state, *args):
        return self.sample(state)


class FakeBelief(BaseBelief):
    STATES = [pomdp_py.SimpleState("love"),
              pomdp_py.SimpleState("hate"),
              pomdp_py.SimpleState("no_feeling")]
    def __init__(self, init_belief="uniform"):
        print("Hello! I am a fake belief.")

    def random(self):
        return pomdp_py.SimpleState(random.sample(FakeBelief.STATES, 1)[0])

    def mpe(self):
        raise NotImplementedError("bye!")

    def __getitem__(self, s):
        return 1.0 / len(FakeBelief.STATES)

    def __iter__(self):
        return iter([pomdp_py.SimpleState(n)
                     for n in FakeBelief.STATES])
