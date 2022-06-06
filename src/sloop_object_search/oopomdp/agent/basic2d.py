import torch
import pomdp_py
from .agent import SloopAgent
from ..domain.transition_model import (StaticObjectTransitionModel,
                                       RobotTransBasic2D)
from sloop.observation import SpatialLanguageObservation

class SloopMosBasic2DAgent(SloopAgent):
    """
    basic --> operates at the fine-grained action level.
    """
    def __init__(self,
                 agent_config,
                 grid_map):
        """
        Args:
            agent_config (dict): various configurations
            grid_map (GridMap): The map to search over. The map
                should have a name.
        """
        self.agent_config = agent_config
        self.grid_map = grid_map
        super().__init__(agent_config, grid_map.map_name)

    def _init_oopomdp(self):
        agent_config = self.agent_config
        grid_map = self.grid_map

        # Prep work
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        action_scheme = agent_config.get("action_scheme", "vw")
        if action_scheme not in {"vw", "xy"}:
            raise ValueError(f"Action scheme {action_scheme} is invalid.")
        no_look = agent_config.get("no_look", True)
        detection_models = {}
        for objid in robot["detectors"]:
            detector_spec = robot["detectors"][objid]
            detection_model = eval(detector_spec[objid]["class"])(
                **detector_spec[objid]["params"]
            )
            detection_models[objid] = detection_model
        search_region = grid_map.free_locations
        init_robot_state = RobotState(robot["id"],
                                      robot["init_pose"],
                                      robot.get("found_objects", tuple()),
                                      robot.get("camera_direction", None))

        # Transition Model
        robot_trans_model = RobotTransBasic2D(
            robot["id"], search_region,
            detection_models, action_scheme)
        transition_models = {robot["id"]: robot_trans_model}
        for objid in objects:
            object_trans_model =\
                eval(objects[objid]["transition"]["class"])(
                    **objects[objid]["transition"].get("params", {})
                )
            transition_models[objid] = object_trans_model
        transition_model = pomdp_py.OOTransitionModel(
            {objid: eval(objects[objid]["transition"]["class"])}
        )

        # Observation Model (Mos)
        observation_model = GMosObservationModel(
            robot["id"], detection_models, no_look=no_look)


        # Policy Model
        target_ids = objects["targets"]
        policy_model = PolicyModelBasic2D(
            robot["id"], target_ids, grid_map, action_scheme,
            observation_model=observation_model)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        # Belief
        target_objects = {objid: objects[objid]
                          for objid in target_ids}
        init_belief = BasicBelief2D(init_robot_state,
                                    target_objects,
                                    search_region,
                                    agent_config["belief"])

        super().__init__(self,
                         init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)


    def update_belief(self, observation, action):
        if isinstance(observation, SpatialLanguageObservation):
            for objid in self.belief.object_beliefs:
                if objid == self.robot_id:
                    new_belief_obj = self.belief.b(self.robot_id)
                else:
                    for


                    self.splang_observation_model.interpret(observation)
        if isinstance()
        self.belief.update(observation, action, self)
