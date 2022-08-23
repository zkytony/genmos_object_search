import pomdp_py
from sloop.agent import SloopAgent
from sloop_object_search.utils.misc import import_class, import_func
from ..models.search_region import SearchRegion2D
from ..models.observation_model import (GMOSObservationModel,
                                        RobotObservationModel,
                                        IdentityLocalizationModel)
from ..models.reward_model import GoalBasedRewardModel
from ..models.transition_model import RobotTransTopo
from . import belief


def init_detection_models(agent_config):
    """A detection model is initialized with nsensor
    and quality parameters.
    """
    robot = agent_config["robot"]

    sensor_specs = {}
    for sensor_spec in robot.get("sensors", []):
        sensor_specs[sensor_spec["name"]] = sensor_spec

    detection_models = {}
    for objid in robot["detectors"]:
        detector_spec = robot["detectors"][objid]
        # Detector spec is either a list of a dictionary.
        # If it is a list, the first element is sensor parameters,
        # while the second element is quality parameters. Note
        # that the sensor parameters could reuse what's specified
        # in robot['sensors']
        if "sensor" in detector_spec["params"]:
            if type(detector_spec["params"]["sensor"]) == str:
                # this is a sensor name
                name = detector_spec["params"]["sensor"]
                sensor_params = sensor_specs[name]["params"]
            else:
                assert type(sensor_params) == dict
                sensor_params = detector_spec["params"]["sensor"]

            detector_params = [sensor_params, detector_spec["params"]["quality"]]
        else:
            detector_params = detector_spec["params"]

        detection_model = import_class(detector_spec["class"])(
            objid, *detector_params
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

def init_object_transition_models(agent_config):
    objects = agent_config["objects"]
    transition_models = {}
    for objid in objects:
        object_trans_model =\
            import_class(objects[objid]["transition"]["class"])(
                objid, **objects[objid]["transition"].get("params", {})
            )
        transition_models[objid] = object_trans_model
    return transition_models

def init_primitive_movements(action_config):
    """
    actions could be configured by:
    - either a list of dicts, {'class': "<action_class>",
                               'params': ... [params to build an action object]}
    - or a single dict {'function': "<function>",
                        'params': ...[params to the function]],
         where the function returns a list of actions.

    The result of parsing action_config should be
    a list of action objects, used as part of the action space during planning.
    """
    if type(action_config) != list\
       and "func" not in action_config:
        raise ValueError("Invalid action config; needs 'func',"
                         "or a list of action specs")
    if type(action_config) == list:
        actions = []
        for action_spec in action_config:
            action = import_class(action_spec["class"])(
                **action_spec["params"])
            actions.append(action)
    else:
        actions = import_func(action_config["func"])(
            **action_config["params"])
    return actions


class MosAgent(pomdp_py.Agent):
    """The top-level class for a mult-object search agent.
    The action space and transition model are not specified here."""
    def __init__(self, agent_config, search_region,
                 init_robot_pose_dist,
                 init_object_beliefs=None):
        """
        Args:
            agent_config (dict): configuration for the agent
            search_region (SearchRegion2D): 2d search region
            init_robot_pose_dist (pomdp_py.GenerativeDistribution):
                belief over initial robot pose
            init_object_beliefs (dict): maps from object id
                to pomdp_py.GenerativeDistribution
        """

        self.agent_config = agent_config
        self.search_region = search_region
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        self.no_look = agent_config.get("no_look", True)
        self.robot_id = robot['id']
        self.target_objects = {target_id: objects[target_id]
                               for target_id in self.agent_config["targets"]}

        # Belief
        init_belief = self.init_belief(
            init_robot_pose_dist, init_object_beliefs)

        # Observation Model (Mos)
        self.detection_models = self.init_detection_models()
        self.localization_model = interpret_localization_model(robot)
        self.robot_observation_model = RobotObservationModel(
            robot['id'], localization_model=self.localization_model)
        observation_model = GMOSObservationModel(
            robot["id"], self.detection_models,
            robot_observation_model=self.robot_observation_model,
            no_look=self.no_look)

        # Reward model
        target_ids = self.agent_config["targets"]
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Transition and policy models
        transition_model, policy_model = self.init_transition_and_policy_models()

        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

    def init_detection_models(self):
        detection_models = init_detection_models(self.agent_config)
        return detection_models

    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None):
        """Override this method if initial belief is constructed in
        a specialized way."""
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                prior=self.agent_config["belief"].get("prior", {}))
        init_robot_belief = belief.init_robot_belief(self.agent_config["robot"],
                                                     init_robot_pose_dist)
        init_belief = pomdp_py.OOBelief({self.robot_id: init_robot_belief,
                                         **init_object_beliefs})
        return init_belief

    def init_transition_and_policy_models(self):
        raise NotImplementedError()

    def reachable(self, pos):
        """Returns True if the given position (as in a viewpoint)
        is reachable by this agent."""
        raise NotImplementedError()

    def update_belief(self, observation, action=None):
        raise NotImplementedError()


class SloopMosAgent(SloopAgent):
    def __init__(self, agent_config, search_region, init_robot_pose_dist,
                 init_object_beliefs=None):
        if not isinstance(search_region, SearchRegion2D):
            raise TypeError("SloopMosAgent requires 2D search region"
                            "because spatial language currently works in 2D.")
        map_name = search_region.grid_map.name
        self.search_region = search_region
        super().__init__(agent_config, map_name,
                         init_robot_pose_dist=init_robot_pose_dist,
                         init_object_beliefs=init_object_beliefs)

    def _init_oopomdp(self, init_robot_pose_dist=None, init_object_beliefs=None):
        raise NotImplementedError()

    def update_belief(self, observation, action=None):
        raise NotImplementedError()
