import pomdp_py
from sloop_object_search.utils.misc import import_class
from ..models.search_region import SearchRegion2D
from ..models.observation_model import (GMOSObservationModel,
                                        RobotObservationModel,
                                        IdentityLocalizationModel)
from ..models.reward_model import GoalBasedRewardModel
from . import belief


class Mos2DAgent(pomdp_py.Agent):
    """The top-level class for 2D agent. A 2D agent is
    one who believes objects lie on a 2D plane, and it
    carries a 2D sensor (e.g. fan-shaped sensor).

    The action space and transition model are not specified here."""
    def __init__(self, agent_config, search_region,
                 init_robot_belief,
                 init_object_beliefs=None):
        """
        Args:
            agent_config (dict): configuration for the agent
            search_region (SearchRegion2D): 2d search region
            init_robot_belief (pomdp_py.GenerativeDistribution): belief over robot state
            init_object_beliefs (dict): maps from object id to pomdp_py.GenerativeDistribution
        """
        assert isinstance(search_region, SearchRegion2D),\
            "search region of a 2D agent should of type SearchRegion2D."
        self.agent_config = agent_config
        self.search_region = search_region
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        no_look = agent_config.get("no_look", True)
        self.robot_id = robot['id']
        self.target_objects = {target_id: objects[target_id]
                               for target_id in self.agent_config["targets"]}

        # Belief
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                prior=self.agent_config["belief"].get("prior", {}))
        init_belief = pomdp_py.OOBelief({robot_id: init_robot_belief,
                                         **init_object_beliefs})

        # Observation Model (Mos)
        detection_models = init_detection_models(agent_config)
        localization_model = interpret_localization_model(robot)
        robot_observation_model = RobotObservationModel(
            robot['id'], localization_model=localization_model)
        observation_model = GMOSObservationModel(
            robot["id"], detection_models,
            robot_observation_model=robot_observation_model,
            no_look=no_look)

        # Reward model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Transition and policy models
        transition_model = self.initialize_transition_model()
        policy_model = self.initialize_policy_model()

        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

    def initialize_transition_model(self):
        raise NotImplementedError()

    def initialize_policy_model(self):
        raise NotImplementedError()


def init_detection_models(agent_config):
    robot = agent_config["robot"]
    detection_models = {}
    for objid in robot["detectors"]:
        detector_spec = robot["detectors"][objid]
        detection_model = import_class(detector_spec["class"])(
            objid, *detector_spec["params"]
        )
        detection_models[objid] = detection_model
    return detection_models

def interpret_localization_model(robot_config):
    """observation model of robot's own state"""
    localization_model = robot_config.get("localization_model", "identity")
    if localization_model == "identity":
        return IdentityLocalizationModel()
    else:
        return import_class(localization_model)(
            **robot_config.get("localization_model_args", {}))
