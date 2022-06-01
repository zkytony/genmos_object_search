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
class TargetObjectState(pomdp_py.ObjectState):
    """A target object's state; The target object is located
    at a location 'loc', with an ID, and a class.

    Note that we are being specific about "target object"
    because in general, you don't have to require an object
    location to be at a single location; the object could be
    represented differently."""
    def __init__(self, objid, objclass, loc):
        super().__init__(objclass, {"loc": loc, "id": objid})

    def __hash__(self):
        return hash((self.id, self.loc))

    @property
    def loc(self):
        return self['loc']

    @property
    def id(self):
        return self['id']

    def copy(self):
        return TargetObjectState(self.id,
                                 self.objclass,
                                 self.loc)


class RobotState(pomdp_py.ObjectState):
    """
    A generic robot state, not assuming particular
    representation of its components. However, we
    do assume that the robot carries a camera and
    can point it to a certain direction. A robot state
    is defined by:

       the robot's id
       the robot pose
       objects found
       camera_direction

    """
    def __init__(self,
                 robot_id,
                 pose,
                 objects_found,
                 camera_direction):
        """
        Note: camera_direction is assumed to be None unless the robot is
        looking at a direction, in which case camera_direction is the string
        e.g. look+x, or 'look'
        """
        super().__init__("robot",
                         {"id": robot_id,
                          "pose": pose,
                          "objects_found": objects_found,
                          "camera_direction": camera_direction})

    def __str__(self):
        return "{}({}, {})".format(self.__class__, self.pose, self.status)

    def __hash__(self):
        return hash(self.pose)

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def objects_found(self):
        return self.attributes['objects_found']

    @property
    def id(self):
        return self.attributes['id']

    @property
    def loc(self):
        """the location of the robot, regardless of orientation"""
        raise NotImplementedError

    def in_range(self, sensor, loc):
        raise NotImplementedError

    @staticmethod
    def from_obz(robot_obz):
        """
        robot_obz (RobotObservation)
        """
        raise NotImplementedError


class JointState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)

    def __str__(self):
        return "{}({})".format(self.__class__, self.object_states)

    def __repr__(self):
        return str(self)

    def loc_of(self, objid):
        return self.object_loc(objid)
