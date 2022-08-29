"""
GMOS observation breaks down to:
- object detection
- robot observation about itself

Note that for all observations that contain
"""
import pomdp_py
import numpy as np
from sloop_object_search.utils.misc import det_dict_hash
from sloop_object_search.utils import math as math_utils

# tolerance of position in object detection / robot localization
# to regard two object detections to be equal. POMDP frame scale.
# Used for planning.
ROBOT_POS_TOL = 2.5
OBJECT_POS_TOL = 2.5
# tolerance of rotation angle in robot localization
# to regard two object detections to be equal.
# Used for planning. Degrees.
ROBOT_ROT_TOL = 15
OBJECT_ROT_TOL = 15

def loc_eq_approx(loc1, loc2, tol=1e-3):
    return math_utils.euclidean_dist(loc1, loc2) <= tol

def get_pos_rot(pose, is_3d=True):
    """Given a pose that could pose1, pose2 could each be either a single tuple, or
    a 2-tuple (position, orientation) where the orientation, return
    a tuple position, orientation. The pose may not contain orientation (i.e. None).
    If 3D, the rotation should be a quaternion tuple qx, qy, qz, qw."""
    if len(pose) == 2 and type(self.pose[0]) == tuple:
        # pose = (position_orientation). Just directly return
        return pose
    else:
        # pose is a single tuple
        if is_3d:
            if len(pose) == 7:
                return pose[:3], pose[3:]
            elif len(pose) == 3:
                return pose, None
            else:
                raise ValueError(f"Invalid dimension of 3D pose: {len(pose)}")
        else:
            if len(pose) == 3:
                return pose[:2], pose[2]
            elif len(pose) == 2:
                return pose, None
            else:
                raise ValueError(f"Invalid dimension of 2D pose: {len(pose)}")

def pose_eq_approx(pose1, pose2, tol_pos=1e-3, tol_rot=1e-3, is_3d=True):
    """
    It's ok for pose1 and/or pose2 to be None. They are equal
    if both are None.
    """
    if pose1 is None:
        return pose2 is None
    if pose2 is None:
        return pose1 is None
    pos1, rot1 = get_pos_rot(pose1, is_3d)
    pos2, rot2 = get_pos_rot(pose1, is_3d)
    pos_eq = loc_eq_approx(pos1, pos2, tol_pos)
    if not pos_eq:
        return False
    if is_3d:
        qdiff_angle_rad = math_utils.quat_diff_angle_relative(pose1[3:], pose2[3:])
        rot_eq = math_utils.to_deg(qdiff_angle_rad) <= tol_rot
        return rot_eq
    else:
        return math_utils.angle_diff_relative(pose1[2], pose2[2]) <= tol_rot


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
        return f"ObjectDetection[{self.objid}]({self.pose}, {self.sizes})"

    def __hash__(self):
        if self.loc is None:
            loc_hash = hash(None)
        else:
            loc_hash = hash(self.loc[i] // OBJECT_POS_TOL for i in self.loc)
        return hash((self.objid, loc_hash))

    def __eq__(self, other):
        if isinstance(other, ObjectDetection):
            return self.objid == other.objid\
                and pose_eq_approx(self.pose, other.pose,
                                   OBJECT_POS_TOL, OBJECT_ROT_TOL,
                                   is_3d=self.is_3d)

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
    def is_3d(self):
        return not self.is_2d

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
    """This is equal to a pose tuple if it is equal to the mean"""
    def __init__(self, robot_id, robot_pose, cov=None):
        """cov: covariance matrix for the robot pose observation."""
        if cov is None:
            cov = np.zeros((len(robot_pose), len(robot_pose)))
        self._cov = cov
        data = (robot_id, robot_pose)
        super().__init__(data)

    @property
    def is_2d(self):
        return len(self.pose) == 3  # x, y, th

    @property
    def is_3d(self):
        return not self.is_2d

    @property
    def cov(self):
        return self._cov

    @property
    def covariance(self):
        return self.cov

    @property
    def pose(self):
        return self.data[1]

    @property
    def robot_id(self):
        return self.data[0]

    @property
    def loc(self):
        if self.is_2d:
            return self.pose[:2]
        else:
            # 3d
            return self.pose[:3]

    def __hash__(self):
        return hash((self.robot_id,
                     (self.loc[i] // POS_TOL
                      for i in self.loc)))

    def __eq__(self, other):
        """This comparison is based on pose approximate equality,
        intended to decrease observation space during planning"""
        if isinstance(other, RobotLocalization):
            return self.robot_id == other.robot_id\
                and pose_eq_approx(self.pose, other.mean,
                                   ROBOT_POS_TOL, ROBOT_ROT_TOL,
                                   is_3d=self.is_3d)
        elif isinstance(other, tuple):
            return pose_eq_approx(self.pose, other,
                                  ROBOT_POS_TOL, ROBOT_ROT_TOL,
                                  is_3d=self.is_3d)
        else:
            return False

    def __str__(self):
        return f"RobotLocalization[{self.robot_id}]({self.pose}, {self.cov})"

    def __repr__(self):
        return str(self)



class RobotObservation(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose_est, objects_found, camera_direction, *args):
        """
        'robot_pose_est' should be a RobotLocalization, if this observation
        is created based on real robot observation. It should be a tuple if
        this observation is generated during planning based on a robot state.
        """
        data = (robot_id, robot_pose_est, objects_found,
                camera_direction, *args)
        super().__init__(data)

    def __str__(self):
        return f"{self.__class__.__name__}[self.robot_id]({self.pose_estimate, self.camera_direction, self.objects_found})"

    def __repr__(self):
        return str(self)

    @property
    def robot_id(self):
        return self.data[0]

    @property
    def pose(self):
        """Returns the most likely robot pose - will return the pose tuple."""
        if isinstance(self.pose_estimate, RobotLocalization):
            return self.pose_estimate.pose
        else:
            return self.pose_estimate

    @property
    def pose_estimate(self):
        return self.data[1]

    @property
    def objects_found(self):
        return self.data[2]

    @property
    def camera_direction(self):
        return self.data[3]

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
                                pose if pose is not None else srobot['pose'],
                                srobot['objects_found'],
                                srobot['camera_direction'])


class RobotObservationTopo(RobotObservation):
    def __init__(self, robot_id, robot_pose_est, objects_found, camera_direction, topo_nid):
        super().__init__(robot_id,
                         robot_pose_est,
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
        objzstrs = [str(self._objobzs[objid])
                    for objid in self._objobzs]
        return "{}(\n  {})".format(self.__class__.__name__, ",\n  ".join(objzstrs))


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
    # a dummy voxel pose used to indicate no detection is made
    NO_POSE = (float('inf'), float('inf'), float('inf'))
    def __init__(self, pose, label):
        """voxel pose could be (x,y,z), which is assumed to be
        at ground resolution level, or (x,y,z,r) which is a location
        at resolution r."""
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
        return "Voxel({}, {})".format(self.pose, self.label)

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


class ObjectVoxel(pomdp_py.SimpleObservation):
    """Optionally, if objid is provided, then this Voxel can
    be thought of as coming from the FOV for the specific
    object, V_objid."""
    def __init__(self, objid, pose, label):
        super().__init__((objid, Voxel(pose, label)))

    @property
    def id(self):
        return self.data[0]

    @property
    def objid(self):
        return self.id

    @property
    def voxel(self):
        return self.data[1]

    @property
    def pose(self):
        return self.voxel.pose

    @property
    def loc(self):
        return self.voxel.loc

    @property
    def label(self):
        return self.voxel.label

    @label.setter
    def label(self, val):
        self.voxel.label = val

    @property
    def res(self):
        return self.voxel.res

    def __str__(self):
        return f"ObjectVoxel[{self.objid}]({self.pose}, {self.label})"


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
