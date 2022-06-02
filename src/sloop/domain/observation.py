"""
In SLOOP, the robot can observe about itself
and about objects. For objects, the observations
are going to be object id and locations.
"""
import pomdp_py

class ObjectObservation(pomdp_py.SimpleObservation):
    def __init__(self, objid, data):
        self.objid = objid
        self.data = data
        super().__init__((objid, data))


class RobotObservation(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, data):
        self.robot_id = robot_id
        self.data = data
        super().__init__((self.robot_id, data))


class JointObservation(pomdp_py.OOObservation):
    """Object-Oriented Observation"""
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
        return "OOObservation({})".format(objzstr)

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
