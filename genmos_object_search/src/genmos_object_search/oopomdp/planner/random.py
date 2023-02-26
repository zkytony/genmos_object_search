import pomdp_py
import random
from genmos_object_search.oopomdp.domain.action import FindAction, Find
from genmos_object_search.oopomdp.domain.observation import Voxel


class RandomPlanner(pomdp_py.Planner):
    """Outputs action uniformly at random. The default
    behavior will exclude Find in the uniform sampling,
    and will take Find when a target was observed in
    the observation upon update."""
    def __init__(self, _, seed=None):
        self._rand = random.Random(seed)
        self._find_next = False

    def plan(self, agent):
        if self._find_next:
            self._find_next = False
            return Find
        all_actions = agent.policy_model.get_all_actions(state=agent.belief.mpe())
        candidate_actions = set(a for a in all_actions if not isinstance(a, FindAction))
        return self._rand.sample(candidate_actions, 1)[0]

    def update(self, agent, action, observation):
        objects_found = set(agent.belief.mpe().s(agent.robot_id).objects_found)
        for objid in agent.target_objects:
            if objid not in objects_found:
                if observation.z(objid).pose != Voxel.NO_POSE:
                    self._find_next = True
                    return
