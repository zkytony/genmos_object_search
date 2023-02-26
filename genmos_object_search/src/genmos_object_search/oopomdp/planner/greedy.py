import pomdp_py
import random
from genmos_object_search.oopomdp.domain.action import FindAction, Find
from genmos_object_search.oopomdp.domain.observation import Voxel
from genmos_object_search.utils.math import euclidean_dist


class GreedyPlanner(pomdp_py.Planner):
    """Outputs action that moves the robot closer to
    the object location with highest belief. The default
    behavior will take Find when a target was observed in
    the observation upon update."""
    def __init__(self, _, **planner_params):
        self._find_next = False

    def plan(self, agent):
        if self._find_next:
            self._find_next = False
            return Find
        mpe_state = agent.belief.mpe()
        all_actions = agent.policy_model.get_all_actions(state=mpe_state)
        candidate_actions = set(a for a in all_actions if not isinstance(a, FindAction))

        objects_found = set(mpe_state.s(agent.robot_id).objects_found)
        if len(objects_found) < len(agent.target_objects):
            target_loc = None
            for objid in agent.target_objects:
                if objid not in objects_found:
                    target_loc = mpe_state.s(objid).loc
                    break
            next_best_action = None
            closest_distance = float('inf')
            for action in candidate_actions:
                next_robot_loc = agent.policy_model.robot_trans_model.sample(
                    mpe_state, action)["pose"][:3]
                target_distance = euclidean_dist(next_robot_loc, target_loc)
                if target_distance < closest_distance:
                    closest_distance = target_distance
                    next_best_action = action
            return next_best_action
        else:
            return random.sample(candidate_actions, 1)[0]

    def update(self, agent, action, observation):
        objects_found = set(agent.belief.mpe().s(agent.robot_id).objects_found)
        for objid in agent.target_objects:
            if objid not in objects_found:
                if observation.z(objid).pose != Voxel.NO_POSE:
                    self._find_next = True
                    return
