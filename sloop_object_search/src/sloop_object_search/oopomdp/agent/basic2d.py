# basic2d: 2D search agent with primitive action space.
import pomdp_py
from ..models.search_region import SearchRegion2D
from ..models.transition_model import RobotTransBasic2D
from ..models.policy_model import PolicyModelBasic2D
from .common import MosAgent, SloopMosAgent, init_object_transition_models, init_primitive_movements

class MosAgentBasic2D(MosAgent):

    def init_transition_and_policy_models(self):
        # transition model
        trans_args = self.agent_config["robot"].get("transition", {})
        robot_trans_model = RobotTransBasic2D(
            self.robot_id, self.detection_models,
            self.reachable, self.no_look, **trans_args
        )
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # policy model
        target_ids = self.agent_config["targets"]
        action_config = self.agent_config["robot"]["action"]
        primitive_movements = init_primitive_movements(action_config)
        policy_model = PolicyModelBasic2D(
            target_ids, robot_trans_model, primitive_movements)
        return transition_model, policy_model

    def reachable(self, pos):
        return pos not in self.search_region.grid_map.obstacles
