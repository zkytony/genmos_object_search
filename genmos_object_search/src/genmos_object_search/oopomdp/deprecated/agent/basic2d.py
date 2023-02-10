import pomdp_py
from sloop.agent import SloopAgent
from sloop.observation import SpatialLanguageObservation
from genmos_object_search.utils.osm import osm_map_to_grid_map
from genmos_object_search.utils.misc import import_class, import_func
from genmos_object_search.oopomdp.domain.state import RobotState
from genmos_object_search.oopomdp.domain import action
from genmos_object_search.oopomdp.models.transition_model import StaticObjectTransitionModel
from genmos_object_search.oopomdp.models.observation_model import (GMOSObservationModel,
                                                                  RobotObservationModel)
from genmos_object_search.oopomdp.models.policy_model import PolicyModelBasic2D
from genmos_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from genmos_object_search.oopomdp.deprecated.models.belief import BeliefBasic2D
from genmos_object_search.oopomdp.deprecated.models.transition_model import RobotTransBasic2D


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


class MosBasic2DAgent(pomdp_py.Agent):
    def __init__(self, agent_config, grid_map, init_belief=None):
        self.agent_config = agent_config

        # Prep work
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        movement_config = robot["primitive_moves"]
        no_look = agent_config.get("no_look", True)
        detection_models = init_detection_models(agent_config)
        search_region = grid_map.filter_by_label("search_region")
        self.robot_id = robot['id']

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
        robot_observation_model = RobotObservationModel(robot['id'])
        observation_model = GMOSObservationModel(
            robot["id"], detection_models,
            robot_observation_model=robot_observation_model,
            no_look=no_look)

        # Policy Model
        target_ids = agent_config["targets"]
        primitive_movements = init_primitive_movements(movement_config)
        policy_model = PolicyModelBasic2D(target_ids,
                                          robot_trans_model,
                                          primitive_movements)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Belief
        if init_belief is None:
            init_robot_state = RobotState(robot["id"],
                                          robot["init_pose"],
                                          robot.get("objects_found", tuple()),
                                          robot.get("camera_direction", None))

            target_objects = {objid: objects[objid]
                              for objid in target_ids}
            init_belief = BeliefBasic2D(target_objects,
                                        belief_config=agent_config["belief"],
                                        robot_state=init_robot_state,
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
    def _init_oopomdp(self, grid_map=None):
        if grid_map is None:
            # No grid map is provided. For now, we assume self.map_name is an OSM map
            self.grid_map = osm_map_to_grid_map(
                self.mapinfo, self.map_name)
        else:
            grid_map = grid_map

        mos_agent = MosBasic2DAgent(self.agent_config, self.grid_map)
        return (mos_agent.belief,
                mos_agent.policy_model,
                mos_agent.transition_model,
                mos_agent.observation_model,
                mos_agent.reward_model)


    def sensor(self, objid):
        return self.observation_model.detection_models[objid].sensor
