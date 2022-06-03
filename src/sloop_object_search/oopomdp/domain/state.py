"""
GMOS State. Factored into object states and robot state.
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
    A specialization of SLOOP's RobotState
    but still general for the multi-object search
    task - assuming particular representation of its
    components. However, for MOS, we do assume that the
    robot carries a camera and can point it to a certain
    direction. A robot state is defined by:

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
