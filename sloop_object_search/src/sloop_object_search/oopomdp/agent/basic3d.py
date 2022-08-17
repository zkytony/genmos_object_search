from . import belief
from .common import MosAgent, SloopMosAgent, init_object_transition_models

class MosAgentBasic3D(MosAgent):
    def init_belief(self, init_robot_pose_dist, init_object_beliefs=None):
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                prior=self.agent_config["belief"].get("prior", {}))
