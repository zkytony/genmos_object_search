from . import belief
from .common import MosAgent, SloopMosAgent, init_object_transition_models

class MosAgentBasic3D(MosAgent):

    def init_transition_and_policy_models(self):
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # TODO: allow for sampling local search viewpoints
        target_ids = self.agent_config["targets"]
        action_config = self.agent_config["robot"]["action"]
        primitive_movements = init_primitive_movements(action_config)
        policy_model = PolicyModelBasic3D(target_ids,
                                          robot_trans_model,
                                          primitive_movements)
        return transition_model, policy_model

    def reachable(self, pos):
        """A position is reachable if it is a valid
        voxel and it is not occupied. Assume 'pos' is a
        position at the ground resolution level"""
        return self.search_region.octree.valid_voxel(*pos, 1)\
            and not self.search_region.occupied_at(pos, res=1)
