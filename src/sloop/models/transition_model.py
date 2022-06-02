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


class ObjectTransitionModel(pomdp_py.TransitionModel):
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
            return 1.0 - 1e-12
        else:
            return 1e-12

    def sample(self, state, action):
        """
        Args:
            state (JointState)
        Returns:
            ObjectState
        """
        return state.s(self.objid).copy()
