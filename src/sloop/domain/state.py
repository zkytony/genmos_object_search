"""
SLOOP's state is factored by objects and a map.

  s = (s1, ..., sn, sr, M)

Note that for efficiency, we will not include
the map actually in the state itself (since
the map is assumed to not change). It will be
an attribute the agent maintains.

Note that the original SLOOP defines its state
specifically in 2D, which is not generally
necessary. Therefore, we define the robot state
slightly more generically but follows the original
SLOOP and MOS3D's idea.
"""
import pomdp_py
class ObjectState(pomdp_py.ObjectState):
    """A object's state; The target object has
    a pose, with an ID, and a class."""
    def __init__(self, objid, objclass, pose):
        super().__init__(objclass, {"pose": pose, "id": objid})

    def __hash__(self):
        return hash((self.id, self.pose))

    @property
    def pose(self):
        return self['pose']

    @property
    def id(self):
        return self['id']

    def copy(self):
        return ObjectState(self.id,
                           self.objclass,
                           self.pose)


class RobotState(pomdp_py.ObjectState):
    """
    A generic robot state, not assuming particular
    representation of its components.
    """
    def __init__(self,
                 robot_id,
                 attributes):
        super().__init__("robot",
                         {"id": robot_id,
                          **attributes})
