import pomdp_py

@dataclass
class Models:
    transition_model: pomdp_py.TransitionModel
    observation_model: pomdp_py.ObservationModel
    reward_model: pomdp_py.RewardModel
    policy_model: pomdp_py.PolicyModel
