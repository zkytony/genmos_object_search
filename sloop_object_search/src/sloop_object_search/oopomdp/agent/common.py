import logging
import pomdp_py
import numpy as np
from genmos_object_search.utils.misc import import_class, import_func
from ..domain.observation import RobotLocalization, RobotObservation, JointObservation
from ..domain.state import RobotState
from ..models.search_region import SearchRegion2D, SearchRegion3D
from ..models.observation_model import (GMOSObservationModel,
                                        RobotObservationModel,
                                        IdentityLocalizationModel)
from ..models.reward_model import GoalBasedRewardModel
from ..models.transition_model import RobotTransTopo
from ..models import belief


def init_detection_models(agent_config):
    """A detection model is initialized with n sensor
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
                sensor_params = detector_spec["params"]["sensor"]
                assert type(sensor_params) == dict

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


def init_visualizer2d(visualizer_class, agent_config, colors=None, **kwargs):
    """initialize 2D visualizer. See utils.visual2d;
    'colors' is mapping from object id (including robot id) to
    an (r, g, b), values ranging 0-255. Since 'agent_config'
    specifies the configuration for one agent, 'colors' could
    store the colors of other agents, which could support visualizing
    multiple agents."""
    # get colors
    if colors is None:
        colors = {}
    for objid in agent_config["objects"]:
        objspec = agent_config["objects"][objid]
        if "color" in objspec:
            colors[objid] = np.asarray(objspec["color"][:3]) * 255
    robot_id = agent_config["robot"]["id"]
    if "color" in agent_config["robot"]:
        colors[robot_id] = np.asarray(agent_config["robot"]["color"][:3]) * 255
    kwargs["colors"] = colors
    return visualizer_class(**kwargs)


class MosAgent(pomdp_py.Agent):
    """The top-level class for a mult-object search agent.
    The action space and transition model are not specified here."""
    def __init__(self, agent_config, search_region,
                 init_robot_pose_dist,
                 init_object_beliefs=None,
                 args_init_object_beliefs=None):
        """
        Args:
            agent_config (dict): configuration for the agent
            search_region (SearchRegion2D): 2d search region
            init_robot_pose_dist (RobotLocalization):
                belief over initial robot pose
            init_object_beliefs (dict): maps from object id
                to pomdp_py.GenerativeDistribution
        """
        if args_init_object_beliefs is None:
            args_init_object_beliefs = {}

        self.agent_config = agent_config
        agent_type = self.agent_config.get("agent_type", "local")
        if agent_type not in {"local", "hierarchical", "local_hierarchical"}:
            raise ValueError("'agent_type' in agent_config must be 'local', 'hierarchical', or 'local_hierarchical'")
        self.agent_type = agent_type
        self.search_region = search_region
        robot_config = agent_config["robot"]
        self.no_look = agent_config.get("no_look", True)
        self.robot_id = robot_config['id']
        self.target_objects = {target_id: agent_config["objects"][target_id]
                               for target_id in self.agent_config["targets"]}

        # Belief
        logging.info("initializing belief")
        init_belief = self.init_belief(
            init_robot_pose_dist, init_object_beliefs,
            **args_init_object_beliefs)

        # Observation Model (Mos)
        self.detection_models = self.init_detection_models()
        localization_model = interpret_localization_model(robot_config)
        self.robot_observation_model = self.init_robot_observation_model(localization_model)
        observation_model = GMOSObservationModel(
            self.robot_id, self.detection_models,
            robot_observation_model=self.robot_observation_model,
            no_look=self.no_look)

        # Reward model
        target_ids = self.agent_config["targets"]
        reward_model = GoalBasedRewardModel(target_ids, robot_id=self.robot_id)

        # Transition and policy models
        transition_model, policy_model = self.init_transition_and_policy_models()

        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

    @property
    def robot_transition_model(self):
        return self.policy_model.robot_trans_model

    def init_robot_observation_model(self, localization_model):
        robot_observation_model = RobotObservationModel(
            self.robot_id, localization_model=localization_model)
        return robot_observation_model

    def init_detection_models(self):
        detection_models = init_detection_models(self.agent_config)
        return detection_models

    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None, **args_init_object_beliefs):
        """Override this method if initial belief is constructed in
        a specialized way."""
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                belief_config=self.agent_config["belief"],
                **args_init_object_beliefs)
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

    @property
    def is_3d(self):
        """Returns True if the agent is 3D; Note that a 3D agent should receive 3D
        object detections and 3D robot pose estimations, while a 2D agent should receive
        2D object detections and 2D robot pose estimations."""
        raise NotImplementedError()

    @property
    def is_hierarchical(self):
        return self.agent_type == "hierarchical"

    @property
    def is_local_hierarchical(self):
        return self.agent_type == "local_hierarchical"

    @property
    def is_local(self):
        return self.agent_type == "local"

    @property
    def reachable_config(self):
        return self.agent_config["robot"].get("reachable", {})

    @property
    def policy_config(self):
        return self.agent_config["robot"]["action"].get("policy", {})

    def update_belief(self, observation, action=None, debug=False, **kwargs):
        """
        update belief given observation.  We can accept two kinds of observations:
        either JointObservation, which should contain object detections and a
        robot observation, or a RobotObservation object, which is used to
        update the robot belief only.

        For RobotObservation, it should contain 'robot_pose' of type
        RobotLocalization.

        This is the single API call that should be used to update the agent's belief.

        kwargs:
            return_fov: if observation contains object detection, then if
               True, return a mapping from objid to (visible_volume, obstacles_hit)

        """
        if isinstance(observation, JointObservation):
            # update robot belief
            if self.robot_id in observation:
                robot_observation = observation.z(self.robot_id)
                if not isinstance(robot_observation, RobotObservation):
                    raise ValueError("For robot belief update, expect robot_observation in observation"\
                                     " to be a RobotObservation.")
                if not isinstance(robot_observation.pose_estimate, RobotLocalization):
                    raise ValueError("For robot belief update, expect pose in robot observation"\
                                     " to be a RobotLocalization which captures uncertainty")
                self._update_robot_belief(robot_observation, action=action, debug=debug, **kwargs)

            # then update object beliefs
            return self._update_object_beliefs(
                observation, action=action, debug=debug, **kwargs)
        elif isinstance(observation, RobotObservation):
            if not isinstance(observation.pose_estimate, RobotLocalization):
                raise ValueError("For robot belief update, expect pose in observation"\
                                 " to be a RobotLocalization which captures uncertainty")
            return self._update_robot_belief(observation, action=action, **kwargs)
        raise NotImplementedError()

    def _update_object_beliefs(self, observation, action=None, debug=False, **kwargs):
        """ Override this function for more specific agents."""
        raise NotImplementedError("update object belief is not implemented")

    def _update_robot_belief(self, observation, action=None, **kwargs):
        """ Override this function for more specific agents.

        kwargs can contain 'robot_state_class' or additional state attributes
        used for creating robot state objects."""
        pose_estimate = observation.pose_estimate
        robot_state_class = kwargs.pop("robot_state_class", RobotState)
        new_robot_belief = belief.RobotStateBelief(
            self.robot_id, pose_estimate,
            objects_found=observation.objects_found,
            camera_direction=observation.camera_direction,
            robot_state_class=robot_state_class,
            **kwargs)
        self.belief.set_object_belief(self.robot_id, new_robot_belief)

    @property
    def visual_config(self):
        return self.agent_config.get("misc", {}).get("visual", {})

try:
    from sloop.agent import SloopAgent
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
except ModuleNotFoundError as ex:
    logging.error("Failed to load sloop: {}".format(ex))
