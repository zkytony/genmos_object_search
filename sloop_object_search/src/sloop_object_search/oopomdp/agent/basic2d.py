# basic2d: 2D search agent with primitive action space.
import pomdp_py
from ..models.search_region import SearchRegion2D
from ..models.transition_model import RobotTransBasic2D
from ..models.policy_model import PolicyModelBasic2D
from .common import MosAgent, SloopMosAgent, init_object_transition_models, init_primitive_movements

class MosAgentBasic2D(MosAgent):

    def init_transition_model(self):
        trans_args = self.agent_config["robot"].get("transition", {})
        robot_trans_model = RobotTransBasic2D(
            self.robot_id, self.detection_models,
            self.reachable, self.no_look, **trans_args
        )
        transition_models = {self.robot_id: robot_trans_model,
                             **init_object_transition_models(self.agent_config)}
        return pomdp_py.OOTransitionModel(transition_models)

    def init_policy_model(self):
        target_ids = self.agent_config["targets"]
        action_config = self.agent_config["robot"]["action"]
        primitive_movements = init_primitive_movements(action_config)
        robot_trans_model = self.transition_model.transition_models[self.robot_id]
        policy_model = PolicyModelBasic2D(
            target_ids, robot_trans_model, primitive_movements)
        return policy_model

    def reachable(self, pos):
        return pos not in self.search_region.grid_map.obstacles
