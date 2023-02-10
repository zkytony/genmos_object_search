"""
GMOS observation breaks down to:
- object detection
- robot observation about itself

Note that for all observations that contain
"""
import pomdp_py
import numpy as np
from genmos_object_search.utils.misc import det_dict_hash
from genmos_object_search.utils import math as math_utils


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
        if not self.is_3d:
            raise RuntimeError("bbox is only for 3D object detections")
        if self.pose == ObjectDetection.NULL:
            raise RuntimeError("NULL object detection has no bounding box.")
        center_bbox = (self.loc, *self.sizes)
        if origin_rep:
            return math_utils.centerbox_to_originbox(center_bbox)
        else:
            return center_bbox

    def to_2d(self):
        """converts the detection to 2D"""
        return ObjectDetection(self.objid, self.loc[:2], sizes=self.sizes[:2])


class RobotLocalization(pomdp_py.Gaussian):
    """This wraps pomdp_py Gaussian as the belief over robot pose - could be
    regarded as an observation of the result of localization. Note that we
    represent robot pose in 3D as (x, y, z, qx, qy, qz, qw), while its
    covariance matrix is about the rotations around the axes. This wrapper
    handles the conversion. That is, the mean and the samples will be 7-tuples.
    """
    def __init__(self, robot_id, pose, covariance=None):
        """covariance (array or list): covariance matrix.
            For 3D pose, the variables are [x y z thx thy thz]
            For 2D pose, the variables are [x y yaw]

        note that for 3D pose, the input argument 'pose' should contain
        quaternion.
        """
        self.robot_id = robot_id
        if len(pose) == 7:
            self._is_3d = True
        elif len(pose) == 3:
            self._is_3d = False
        else:
            raise ValueError("Invalid dimension of robot pose estimate. "\
                             f"Expected 3 or 7, but got {len(pose)}")
        if covariance is None:
            covariance = np.zeros((len(robot_pose), len(robot_pose)))
        if isinstance(covariance, np.ndarray):
            covariance = covariance.tolist()
        if self.is_3d:
            super().__init__(list(RobotLocalization._to_euler_pose(pose)), covariance)
        else:
            super().__init__(list(pose), covariance)

    @property
    def is_3d(self):
        return self._is_3d

    @property
    def is_2d(self):
        return not self.is_3d

    @property
    def mean(self):
        if self.is_3d:
            return RobotLocalization._to_quat_pose(super().mean)
        else:
            return tuple(super().mean)

    @property
    def pose(self):
        return self.mean

    @property
    def loc(self):
        if self.is_2d:
            return self.pose[:2]
        else:
            # 3d
            return self.pose[:3]

    @staticmethod
    def _to_euler_pose(pose7tup):
        return (*pose7tup[:3], *math_utils.quat_to_euler(*pose7tup[3:]))

    @staticmethod
    def _to_quat_pose(pose6tup):
        return (*pose6tup[:3], *math_utils.euler_to_quat(*pose6tup[3:]))

    def __getitem__(self, pose):
        if self.is_3d:
            return super().__getitem__(RobotLocalization._to_euler_pose(pose))
        else:
            return super().__getitem__(pose)

    def __mpe__(self):
        pose = tuple(super().mpe())
        if self.is_3d:
            return RobotLocalization._to_quat_pose(pose)
        else:
            return pose

    def random(self):
        pose = tuple(super().random())
        if self.is_3d:
            return RobotLocalization._to_quat_pose(pose)
        else:
            return pose

    def to_2d(self):
        if self.is_3d:
            x, y, _ = self.mean[:3]
            _, _, yaw = math_utils.quat_to_euler(*self.mean[3:])
            cov = np.asarray(self.cov)[np.ix_((0,1,-1), (0,1,-1))]  # get covariance matrix
            return RobotLocalization(self.robot_id, (x,y,yaw), cov)
        # already 2D
        return self

    def __str__(self):
        return f"RobotLocalization[{self.robot_id}]({self.pose}, {self.cov})"

    def __repr__(self):
        return str(self)


class RobotObservation(pomdp_py.SimpleObservation):
    def __init__(self, robot_id, robot_pose_est, objects_found,
                 camera_direction, *args):
        """If this observation is created based on real robot observation,
        'robot_pose_est' should be a RobotLocalization, It should be a tuple if
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
    def pose_est(self):
        return self.pose_estimate

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
    """The equality comparison of RobotObseravtionTopo does not consider
    the robot pose -- comparing the node id and topo map hashcode is
    sufficient. The pose estimation is still useful as a grounding
    of a viewpoint at the node. This observation type is only used
    for sampling observations during planning."""
    def __init__(self, robot_id, robot_pose_est, objects_found,
                 camera_direction, topo_nid, topo_map_hashcode):
        super().__init__(robot_id,
                         robot_pose_est,
                         objects_found,
                         camera_direction,
                         topo_nid,
                         topo_map_hashcode)

    @property
    def topo_nid(self):
        return self.data[-2]

    @property
    def topo_map_hashcode(self):
        return self.data[-1]

    @staticmethod
    def from_state(srobot, pose=None):
        return RobotObservationTopo(srobot['id'],
                                    pose if pose is None else srobot['pose'],
                                    srobot['objects_found'],
                                    srobot['camera_direction'],
                                    srobot['topo_nid'],
                                    srobot['topo_map_hashcode'])

    def __hash__(self):
        return hash((self.robot_id, self.topo_nid, self.topo_map_hashcode,
                     self.objects_found))

    def __eq__(self, other):
        if isinstance(other, RobotObservationTopo):
            return self.robot_id == other.robot_id\
                and self.objects_found == other.objects_found\
                and self.camera_direction == other.camera_direction\
                and self.topo_nid == other.topo_nid\
                and self.topo_map_hashcode == other.topo_map_hashcode
        else:
            return False


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

########### Used for planning for 2D observation sampling #########
class ObjectLoc(pomdp_py.SimpleObservation):
    """Optionally, if objid is provided, then this can
    be thought of as coming from the FOV for the specific
    object, V_objid. Similar to ObjectVoxel but for 2D."""

    UNKNOWN = "unknown"
    FREE = "free"
    OTHER = "other" #i.e. not i (same as FREE but for object observation)
    NO_LOC = (float('inf'), float('inf'))

    def __init__(self, objid, loc, label):
        super().__init__((objid, (loc, label)))

    @property
    def id(self):
        return self.data[0]

    @property
    def objid(self):
        return self.id

    @property
    def loc(self):
        return self.data[1][0]

    @property
    def label(self):
        return self.data[1][1]

    def __str__(self):
        return f"ObjectLoc[{self.objid}]({self.loc}, {self.label})"
