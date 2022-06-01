"""
In SLOOP, the robot can observe about itself
and about objects. For objects, the observations
are going to be object id and locations.
"""
import pomdp_py

class TargetLoc(pomdp_py.SimpleObservation):
    """Observation of a target object's location"""
    NULL = None  # empty
    def __init__(self, objid, loc):
        self.objid = objid
        self.loc = loc
        super().__init__((objid, loc))
    def __str__(self):
        return f"{self.objid}({self.objid}, {self.loc})"
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


class JointObservation(pomdp_py.Observation):
    def __init__(self, robotobz, objobzs):
        """
        objobzs (dict): maps from objid to Loc or NULL
        """
        self._hashcode = hash(frozenset(objobzs.items()))
        if isinstance(robotobz, RobotState):
            robotobz = RobotObservation(robotobz.id,
                                        robotobz['pose'],
                                        robotobz['status'].copy())
        self._robotobz = robotobz
        self._objobzs = objobzs

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self._objobzs == other._objobzs

    def __str__(self):
        robotstr = str(self._robotobz)
        objzstr = ""
        for objid in self._objobzs:
            if self._objobzs[objid].loc is not None:
                objzstr += str(self._objobzs[objid])
        return "JointObservation(r:{};o:{})".format(robotstr, objzstr)

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
        if objid == self._robotobz.robot_id:
            return self._robotobz
        elif objid in self._objobzs:
            return self._objobzs[objid]
        else:
            raise ValueError("Object ID {} not in observation".format(objid))

    @property
    def z_robot(self):
        return self._robotobz

    def has_positive_detection(self):
        return any(zi.loc is not None for zi in self)
