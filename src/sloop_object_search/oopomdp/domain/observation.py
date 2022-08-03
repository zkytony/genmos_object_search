"""
GMOS observation breaks down to:
- object detection
- robot observation about itself
"""
import pomdp_py

class ObjectDetection(pomdp_py.SimpleObservation):
    """Observation of a target object's location"""
    NULL = None  # empty
    NO_POSE = "no_pose"
    def __init__(self, objid, pose):
        super().__init__((objid, pose))

    @property
    def pose(self):
        return self.data[1]

    def __str__(self):
        return f"{self.objid}({self.objid}, {self.pose})"

    @property
    def id(self):
        return self.data[0]

    @property
    def objid(self):
        return self.data[0]

    @staticmethod
    def null_observation(objid):
        return ObjectDetection(objid, ObjectDetection.NULL)

    @property
    def is_2d(self):
        return len(self.loc) == 2

    @property
    def loc(self):
        return self.pose

    @staticmethod
    def null_observation(objid):
        return ObjectDetection(objid, ObjectDetection.NULL)


class RobotObservation(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose, objects_found, camera_direction, *args):
        self.robot_id = robot_id
        self.pose = robot_pose
        self.objects_found = objects_found
        self.camera_direction = camera_direction
        data = (self.robot_id, self.pose, self.objects_found, self.camera_direction, *args)
        super().__init__(data)

    def __str__(self):
        return f"{self.robot_id}({self.pose, self.camera_direction, self.objects_found})"

    @property
    def is_2d(self):
        return len(self.pose) == 3  # x, y, th

    @property
    def loc(self):
        if self.is_2d:
            return self.pose[:2]
        else:
            # 3d
            return self.pose[:3]

    @staticmethod
    def from_state(srobot):
        return RobotObservation(srobot['id'],
                                srobot['pose'],
                                srobot['objects_found'],
                                srobot['camera_direction'])

class RobotObservationTopo(RobotObservation):
    def __init__(self, robot_id, robot_pose, objects_found, camera_direction, topo_nid):
        super().__init__(robot_id,
                         robot_pose,
                         objects_found,
                         camera_direction,
                         topo_nid)

    @property
    def topo_nid(self):
        return self.data[-1]

    @staticmethod
    def from_state(srobot):
        return RobotObservationTopo(srobot['id'],
                                    srobot['pose'],
                                    srobot['objects_found'],
                                    srobot['camera_direction'],
                                    srobot['topo_nid'])


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
        return iter(self._objobzs)

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

################## Voxel, ported over from 3D-MOS ######################
class Voxel:
    FREE = "free"
    OTHER = "other" #i.e. not i (same as FREE but for object observation)
    UNKNOWN = "unknown"
    def __init__(self, pose, label):
        self._pose = pose
        self._label = label
    @property
    def pose(self):
        return self._pose
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, val):
        self._label = val
    def __str__(self):
        if self._pose is None:
            return "(%s, %s)" % (None, self._label)
        else:
            return "(%d, %d, %d, %s)" % (*self._pose, self._label)
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return hash((*self._pose, self._label))
    def __eq__(self, other):
        if not isinstance(other, Voxel):
            return False
        else:
            return self._pose == other.pose\
                and self._label == other.label


class FovVoxels:

    """Voxels in the field of view."""


    def __init__(self, voxels):
        """
        voxels: dictionary (x,y,z)->Voxel, or objid->Voxel
                If this is the unfactored observation, then there are UNKNOWN
                voxels in this set. Otherwise, voxels can either be labeled i or OTHER,
        """
        self._voxels = voxels

    def __contains__(self, item):
        if type(item) == tuple  or type(item) == int:
            return item in self._voxels
        elif isinstance(item, Voxel):
            return item.pose in self._voxels\
                and self._voxels[item.pose].label == item.label
        else:
            return False

    def __getitem__(self, item):
        if item not in self:
            raise ValueError("%s is not contained in this FovVoxels object." % str(item))
        else:
            if type(item) == tuple or type(item) == int:
                return self._voxels[item]
            else:  # Must be Voxel
                return self._voxels[item.pose]

    @property
    def voxels(self):
        return self._voxels

    def __eq__(self, other):
        if not isinstance(other, FovVoxels):
            return False
        else:
            return self._voxels == other.voxels
