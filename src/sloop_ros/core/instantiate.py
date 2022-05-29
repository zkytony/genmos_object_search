import importlib
from sloop_ros.utils.misc import import_class
from .framework import Models


def create_models(config):
    transmodel_class = import_class(config['transition_model'])
    obsrvmodel_class = import_class(config['observation_model'])
    rewardmodel_class = import_class(config['reward_model'])
    policymodel_class = import_class(config['policy_model'])

    transition_model = transmodel_class(**config.get("transition_model_params", {}))
    observation_model = obsrvmodel_class(**config.get("observation_model_params", {}))
    reward_model = rewardmodel_class(**config.get("reward_model_params", {}))
    policy_model = policymodel_class(**config.get("policy_model_params", {}))

    return Models(transition_model,
                  observation_model,
                  reward_model,
                  policy_model)
