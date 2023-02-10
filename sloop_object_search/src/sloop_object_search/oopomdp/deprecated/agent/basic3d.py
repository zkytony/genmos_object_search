
import pomdp_py
from genmos_object_search.oopomdp.domain.state import RobotState, ObjectState
from genmos_object_search.oopomdp.models.observation_model import (GMOSObservationModel,
                                                                  RobotObservationModel)
from genmos_object_search.oopomdp.models.policy_model import PolicyModelBasic3D
from genmos_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from genmos_object_search.oopomdp.deprecated.models.belief import BeliefBasic3D
from genmos_object_search.oopomdp.deprecated.models.transition_model import RobotTransBasic3D

from .basic2d import (init_detection_models,
                      init_object_transition_models,
                      init_primitive_movements)


class MosBasic3DAgent(pomdp_py.Agent):
    def __init__(self, agent_config, init_belief=None):
        # TODO: need a representation of 3D map. Currently,
        # this is only for preliminary testing.
        positions = set((x,y,z)
                        for x in range(-10,10)
                        for y in range(-10,10)
                        for z in range(-10,10))

        self.agent_config = agent_config

        # prep
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        movement_config = robot["primitive_moves"]
        no_look = agent_config.get("no_look", True)
        detection_models = init_detection_models(agent_config)
        self.robot_id = robot['id']

        # Transition Model
        robot_trans_model = RobotTransBasic3D(
            robot["id"], positions,
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
        policy_model = PolicyModelBasic3D(target_ids,
                                          robot_trans_model,
                                          primitive_movements)

        # Reward Model
        reward_model = GoalBasedRewardModel(target_ids, robot_id=robot["id"])

        if init_belief is None:
            init_robot_state = RobotState(robot["id"],
                                          robot["init_pose"],
                                          robot.get("objects_found", tuple()),
                                          robot.get("camera_direction", None))

            target_objects = {objid: objects[objid]
                              for objid in target_ids}
            init_belief = BeliefBasic3D(init_robot_state,
                                        target_objects,
                                        agent_config["belief"])

        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)
