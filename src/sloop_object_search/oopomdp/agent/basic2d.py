import pomdp_py
from ..domain.transition_model import (StaticObjectTransitionModel,
                                       RobotTransBasic2D)

class SloopMosBasic2DAgent(pomdp_py.Agent):
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
        # Prep work
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

        # Transition Model
        robot_trans_model = RobotTransBasic2D(
            robot["id"], grid_map.free_locations,
            detection_models, action_scheme)
        robot = agent_config["robot"]
        objects = agent_config["objects"]
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

        # Observation Model
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
        init_belief = self.initialize_belief(target_ids, grid_map, agent_config["belief"])

        super().__init__(self,
                         init_belief,
                         policy_model
                         transition_model,
                         observation_model,
                         reward_model)
