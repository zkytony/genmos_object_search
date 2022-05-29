from dataclasses import dataclass
import pomdp_py

@dataclass
class Models:
    transition_model: pomdp_py.TransitionModel
    observation_model: pomdp_py.ObservationModel
    reward_model: pomdp_py.RewardModel
    policy_model: pomdp_py.PolicyModel


class BaseAgent(pomdp_py.Agent):
    def __init__(self, belief, models, **kwargs):
        super().__init__(belief,
                         models.policy_model,
                         transition_model=models.transition_model,
                         observation_model=models.observation_model,
                         reward_model=models.reward_model)
        self._planner = None

    def setup(self):
        raise NotImplementedError

    def set_planner(self, planner):
        self._planner = planner
