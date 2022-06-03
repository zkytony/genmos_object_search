"""
GMOS observation breaks down to:
- object detection
- robot observation about itself
"""
import sloop

class ObjectDetection(pomdp_py.SimpleObservation):
    """Observation of a target object's location"""
    NULL = None  # empty
    def __init__(self, objid, pose):
        super().__init__(objid, pose)

    @property
    def pose(self):
        return self.data

    def __str__(self):
        return f"{self.objid}({self.objid}, {self.pose})"

    @property
    def id(self):
        return self.objid


class RobotObservation(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose, objects_found, camera_direction):
        self.robot_id = robot_id
        self.pose = robot_pose
        self.objects_found = objects_found
        self.camera_direction = camera_direction
        super().__init__((self.robot_id, self.pose, self.objects_found, self.camera_direction))

    def __str__(self):
        return f"{self.robot_id}({self.pose, self.camera_direction, self.objects_found})"


class GMOSObservation(pomdp_py.Observation):
    """Joint observation of objects for GMOS"""
    def __init__(self, objobzs):
        """
        objobzs (dict): Maps from object id to Observation.
            (can include robot observation)
        """
        self._hashcode = hash(frozenset(objobzs.items()))
        self._objobzs = objobzs

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self._objobzs == other._objobzs

    def __str__(self):
        objzstr = ""
        for objid in self._objobzs:
            if self._objobzs[objid].loc is not None:
                objzstr += str(self._objobzs[objid])
        return "{}({})".format(self.__class__.__name__, objzstr)

    def __repr__(self):
        return str(self)

    def __len__(self):
        # Only care about object observations here
        return len(self._objobzs)

    def __iter__(self):
        # Only care about object observations here
        return iter(self._objobzs.values())

    def __getitem__(self, objid):
        # objid can be either object id or robot id.
        return self.z(objid)

    def z(self, objid):
        if objid in self._objobzs:
            return self._objobzs[objid]
        else:
            raise ValueError("Object ID {} not in observation".format(objid))

    def __contains__(self, objid):
        return objid in self._objobzs
