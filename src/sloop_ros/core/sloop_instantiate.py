import rospy
from dataclasses import dataclass
import pomdp_py
import importlib

@dataclass
class Models:
    transition_model: pomdp_py.TransitionModel
    observation_model: pomdp_py.ObservationModel
    reward_model: pomdp_py.RewardModel
    policy_model: pomdp_py.PolicyModel


def create_models(config):
    module_name = config["module"]
    module = importlib.import_module(module_name)

    rospy.loginfo(f"{module_name}.{config['transition_model']}")
    transmodel_class = getattr(module, config['transition_model'])

    obsrvmodel_class = getattr(module, config['observation_model'])
    rewardmodel_class = getattr(module, config['reward_model'])
    policymodel_class = getattr(module, config['policy_model'])

    transition_model = transmodel_class(**config.get("transition_model_params", {}))
    observation_model = obsrvmodel_class(**config.get("observation_model_params", {}))
    reward_model = rewardmodel_class(**config.get("reward_model_params", {}))
    policy_model = policymodel_class(**config.get("policy_model_params", {}))

    return (transition_model,
            observation_model,
            reward_model,
            policy_model)
