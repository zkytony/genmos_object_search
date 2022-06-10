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
    robot = agent_config["robot"]
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


class MosBasic2DAgent(pomdp_py.Agent):
    def __init__(self, agent_config, grid_map, init_belief=None):
        self.agent_config = agent_config

        # Prep work
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        action_config = agent_config["action"]
        no_look = agent_config.get("no_look", True)
        detection_models = init_detection_models(agent_config)
        search_region = grid_map.filter_by_label("search_region")

        # Transition Model
        reachable_positions = grid_map.filter_by_label("reachable")
        robot_trans_model = RobotTransBasic2D(
            robot["id"], reachable_positions,
            detection_models,
            no_look=no_look)
        transition_models = {robot["id"]: robot_trans_model,
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
                                          observation_model,
                                          **action_config)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Belief
        if init_belief is None:
            init_robot_state = RobotState2D(robot["id"],
                                            robot["init_pose"],
                                            robot.get("objects_found", tuple()),
                                            robot.get("camera_direction", None))

            target_objects = {objid: objects[objid]
                              for objid in target_ids}
            init_belief = BeliefBasic2D(init_robot_state,
                                        target_objects,
                                        agent_config["belief"],
                                        search_region=search_region,
                                        object_beliefs=agent_config.get("object_beliefs", None))
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

    def update_belief(self, observation, action):
        self.belief.update_robot_belief(observation, action)
        next_robot_state = self.belief.mpe().s(self.robot_id)
        for objid in self.belief.object_beliefs:
            if objid == self.robot_id:
                continue
            else:
                self.belief.update_object_belief(
                    self, objid, observation,
                    next_robot_state, action)



class SloopMosBasic2DAgent(SloopAgent):
    """
    basic --> operates at the fine-grained action level.
    """
    def _init_oopomdp(self):
        self.grid_map = osm_map_to_grid_map(
            self.mapinfo, self.map_name)
        mos_agent = MosBasic2DAgent(self.agent_config, self.grid_map)
        return (mos_agent.belief,
                mos_agent.policy_model,
                mos_agent.transition_model,
                mos_agent.observation_model,
                mos_agent.reward_model)


    def sensor(self, objid):
        return self.observation_model.detection_models[objid].sensor
