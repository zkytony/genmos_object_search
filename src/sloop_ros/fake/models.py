import pomdp_py
import random

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
        return pomdp_py.SimpleObservation(state.data)


class FakeRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        if action.name == state.data:
            return 10
        else:
            return -5

class FakePolicyModel(pomdp_py.RolloutPolicy):
    ACTIONS = {pomdp_py.SimpleAction(n)
               for n in {"left", "right"}}
    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_action(self, state=None, history=None):
        return FakePolicyModel.ACTIONS

    def rollout(self, state, *args):
        return self.sample(state)
