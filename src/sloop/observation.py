import pomdp_py
from pomdp_py.utils.cython_utils import det_dict_hash

class ObjectObservation(pomdp_py.SimpleObservation):
    def __init__(self, objid, pose):
        self.objid = objid
        self.pose = pose
        super().__init__((objid, pose))

    def __str__(self):
        return f"object({self.objid}, {self.pose})"


class JointObservation(pomdp_py.Observation):
    def __init__(self, objobzs):
        self._hashcache = -1
        self._objobzs = objobzs

    def __hash__(self):
        if self._hashcache == -1:
            self._hashcache = det_dict_hash(self._objobzs)
        return self._hashcache

    def __eq__(self, other):
        return self._objobzs == other._objobzs

    def __str__(self):
        objzstr = ""
        for objid in self._objobzs:
            if self._objobzs[objid].loc is not None:
                objzstr += "{}{}".format(objid, self._objobzs[objid].loc)
        return "JointObservation({})".format(objzstr)

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
