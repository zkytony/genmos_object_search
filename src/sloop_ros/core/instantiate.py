import pomdp_py
import importlib
from sloop_ros.utils.misc import import_class
from dataclasses import dataclass

@dataclass
class Models:
    transition_model: pomdp_py.TransitionModel
    observation_model: pomdp_py.ObservationModel
    reward_model: pomdp_py.RewardModel
    policy_model: pomdp_py.PolicyModel


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

def create_agent(belief, models, config):
    agent_class = import_class(config["agent"])
    print(f"Creating Agent of class {agent_class}")
    sloop_agent = agent_class(
        belief, models, **config.get("agent_params", {}))
    return sloop_agent


def create_planner(config):
    planner = import_class(config["planner"])(**config.get("planner_config", {}))
    return planner

def initialize_belief(config):
    belief = import_class(config["belief_dist"])(**config["belief_params"])
    return belief
