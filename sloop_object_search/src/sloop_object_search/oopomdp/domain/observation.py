"""
GMOS observation breaks down to:
- object detection
- robot observation about itself
"""
import pomdp_py
from sloop_object_search.utils.misc import det_dict_hash
from sloop_object_search.utils import math as math_utils

class ObjectDetection(pomdp_py.SimpleObservation):
    """Observation of a target object's location"""
    NULL = None  # empty
    NO_POSE = "no_pose"
    def __init__(self, objid, pose, sizes=None):
        """
        pose: Either a single tuple for position-only,
                or a tuple (position, orientation).
        sizes: a tuple (w, l) or (w, l, h)
            used to indicate the dimensions of the detected object;
            The detection box is centered at 'pose'.
        """
        # let's not compare equality using sizes as they
        # are usually noisy - so we leave it out in super() below
        if sizes is None:
            if pose != ObjectDetection.NULL:
                sizes = (1, 1, 1)
        self._sizes = sizes
        super().__init__((objid, pose))

    @property
    def pose(self):
        return self.data[1]

    def __str__(self):
        return f"{self.objid}({self.objid}, {self.pose}, {self.sizes})"

    @property
    def sizes(self):
        return self._sizes

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
        # location (i.e. position)
        if self.pose is None:
            return None
        if len(self.pose) == 2 and type(self.pose[0]) == tuple:
            position = self.pose[0]
            return position
        else:
            # position-only
            return self.pose

    @staticmethod
    def null_observation(objid):
        return ObjectDetection(objid, ObjectDetection.NULL)

    @property
    def bbox_axis_aligned(self, origin_rep=True):
        """axis-aligned boudning box. If origin_rep is True,
        return origin-based box. Otherwise,
        return center-based box"""
        if self.pose == ObjectDetection.NULL:
            raise RuntimeError("NULL object detection has no bounding box.")
        center_bbox = (self.loc, *self.sizes)
        if origin_rep:
            return math_utils.centerbox_to_originbox(center_bbox)
        else:
            return center_bbox

class RobotLocalization(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose):
        self.robot_id = robot_id
        self.pose = robot_pose
        data = (self.robot_id, self.pose)
        super().__init__(data)

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
    def from_state(srobot, pose=None):
        """If 'pose' is set, it overrides the pose in srobot"""
        return RobotObservation(srobot['id'],
                                pose if pose is None else srobot['pose'],
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
    def from_state(srobot, pose=None):
        return RobotObservationTopo(srobot['id'],
                                    pose if pose is None else srobot['pose'],
                                    srobot['objects_found'],
                                    srobot['camera_direction'],
                                    srobot['topo_nid'])


class GMOSObservation(pomdp_py.Observation):
    """Joint observation of objects for GMOS;
    Note that underneath the hood, object observations
    are independently factored.

    Refer to FovVoxels which is meant to contain
    a set of voxels that represent the unfactored observation
    of multiple objects.
    """
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
# alias
JointObservation = GMOSObservation

################## Voxel, ported over from 3D-MOS ######################
class Voxel(pomdp_py.Observation):
    FREE = "free"
    OTHER = "other" #i.e. not i (same as FREE but for object observation)
    UNKNOWN = "unknown"
    def __init__(self, pose, label):
        """voxel pose could be (x,y,z), which is assumed to be
        at ground resolution level, or (x,y,z,r) which is a location
        at resolution r"""
        self._pose = pose
        self._label = label
    @property
    def pose(self):
        return self._pose
    @property
    def loc(self):
        return self.pose[:3]
    @property
    def res(self):
        if len(self.pose) == 3:
            return 1
        else:
            return self.pose[-1]
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, val):
        self._label = val
    def __str__(self):
        return "({}, {})".format(self.pose, self.label)
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


class FovVoxels(pomdp_py.Observation):

    """Voxels in the field of view.

    Immutable object."""


    def __init__(self, voxels):
        """
        voxels: dictionary (x,y,z)->Voxel, (x,y,z,r)->Voxel, or objid->Voxel
                If this is the unfactored observation, then there are UNKNOWN
                voxels in this set. Otherwise, voxels can either be labeled i or OTHER,
        """
        self._voxels = voxels
        self._hashcache = -1

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

    def __hash__(self):
        if self._hashcache == -1:
            self._hashcache = det_dict_hash(self._voxels)
        else:
            return self._hashcache
