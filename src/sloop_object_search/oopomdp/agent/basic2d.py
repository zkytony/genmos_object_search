import pomdp_py
from sloop.agent import SloopAgent
from sloop.observation import SpatialLanguageObservation
from sloop_object_search.utils.osm import osm_map_to_grid_map
from sloop_object_search.utils.misc import import_class
from ..domain.state import RobotState2D
from ..models.transition_model import (StaticObjectTransitionModel,
                                       RobotTransBasic2D)
from ..models.observation_model import (GMOSObservationModel,
                                        RobotObservationModel2D)
from ..models.policy_model import PolicyModelBasic2D
from ..models.reward_model import GoalBasedRewardModel
from ..models.belief import BeliefBasic2D


def init_detection_models(agent_config):
    detection_models = {}
    for objid in robot["detectors"]:
        detector_spec = robot["detectors"][objid]
        detection_model = import_class(detector_spec["class"])(
            objid, *detector_spec["params"]
        )
        detection_models[objid] = detection_model
    return detection_models

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


class SloopMosBasic2DAgent(SloopAgent):
    """
    basic --> operates at the fine-grained action level.
    """
    def _init_oopomdp(self):
        agent_config = self.agent_config

        self.grid_map = osm_map_to_grid_map(
            self.mapinfo, self.map_name)

        # Prep work
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        action_scheme = agent_config.get("action_scheme", "vw")
        if action_scheme not in {"vw", "xy"}:
            raise ValueError(f"Action scheme {action_scheme} is invalid.")
        no_look = agent_config.get("no_look", True)
        detection_models = init_detection_models(agent_config)
        search_region = self.grid_map.filter_by_label("search_region")
        init_robot_state = RobotState2D(robot["id"],
                                        robot["init_pose"],
                                        robot.get("found_objects", tuple()),
                                        robot.get("camera_direction", None))

        # Transition Model
        reachable_positions = self.grid_map.filter_by_label("reachable")
        robot_trans_model = RobotTransBasic2D(
            robot["id"], reachable_positions,
            detection_models, action_scheme,
            no_look=no_look)
        transition_models = {**{robot["id"]: robot_trans_model},
                             **init_object_transition_models(agent_config)}
        transition_model = pomdp_py.OOTransitionModel(transition_models)

        # Observation Model (Mos)
        robot_observation_model = RobotObservationModel2D(robot['id'])
        observation_model = GMOSObservationModel(
            robot["id"], detection_models,
            robot_observation_model=robot_observation_model,
            no_look=no_look)

        # Policy Model
        target_ids = agent_config["targets"]
        policy_model = PolicyModelBasic2D(target_ids,
                                          robot_trans_model,
                                          action_scheme,
                                          observation_model,
                                          no_look=no_look)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Belief
        target_objects = {objid: objects[objid]
                          for objid in target_ids}
        init_belief = BeliefBasic2D(init_robot_state,
                                    target_objects,
                                    search_region,
                                    agent_config["belief"])

        return (init_belief,
                policy_model,
                transition_model,
                observation_model,
                reward_model)


    def sensor(self, objid):
        return self.observation_model.detection_models[objid].sensor
