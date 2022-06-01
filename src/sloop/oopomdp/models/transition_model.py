"""
SLOOP transition model consists of transition
of the robot and the objects. Objects are by
default static.
"""
import pomdp_py
import math

class RobotTransitionModel(pomdp_py.TransitionModel):
    """Models Pr(sr' | s, a); Likely domain-specific"""
    def __init__(self, robot_id):
        self.robot_id = robot_id


class TargetObjectTransitionModel(pomdp_py.TransitionModel):
    """Models Pr(si' | s, a); By default, it is static;
    Could be extended. Likely domain-specific"""
    def __init__(self, objid):
        self.objid = objid

    def probability(self, next_object_state, state, action):
        """
        Args:
            next_object_state (TargetObjectState): assumes to
                have the same object id as this transition model.
            state (JointState)
            action (pomdp_py.Action)
        Returns:
            float
        """
        if next_object_state == state.s(self.objid):
            return 1.0 - 1e-9
        else:
            return 1e-9

    def sample(self, state, action):
        """
        Args:
            state (JointState)
        Returns:
            TargetObjectState
        """
        return state.s(self.objid).copy()


class JointTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, trans_models):
        """
        Args:
            trans_models (dict): Maps from id to TransitionModel
        """
        self.trans_models = trans_models

    def sample(self, state, action):
        """
        Args:
            state (JointState)
        Returns:
            JointState
        """
        # note that objid could be robot id or target object id.
        return JointState({self.trans_models[objid].sample(state, action)
                           for objid in self.trans_models})

    def probability(self, next_state, state, action):
        """
        Assumes object independence.

        Args:
            next_state (JointState)
            state (JointState)
            action (pomdp_py.Action)
        Returns:
            float
        """
        return math.prod([
            self.trans_models[objid].probability(next_state.s(objid), state, action)
            for objid in self.trans_models])

    def __getitem__(self, objid):
        return self.trans_models[objid]
