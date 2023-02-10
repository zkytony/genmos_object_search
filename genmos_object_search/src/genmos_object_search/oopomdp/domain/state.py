"""
GMOS State. Factored into object states and robot state.
"""
import pomdp_py


class ObjectState(pomdp_py.ObjectState):
    """A object's state; The target object has
    a pose, with an ID, and a class."""
    def __init__(self, objid, objclass, pose, res=None):
        """
        Args:
            objid (str): object id
            objclass (str): object class
            pose (hashable): object pose. Either a single tuple for position-only,
                or a tuple (position, orientation).
            res (None or int): resolution of pose.
                Used by 3D multi-res POMDP. None by default.
        """
        super().__init__(objclass,
                         {"pose": pose,
                          "id": objid,
                          "res": res})

    def __hash__(self):
        return hash((self.id, self.pose, self.res))

    @property
    def pose(self):
        return self['pose']

    @property
    def loc(self):
        """object location; we don't consider orientation for now,
        so pose is the rotation."""
        if len(self.pose) == 2 and type(self.pose[0]) == tuple:
            position = self.pose[0]
            return position
        else:
            # position-only
            return self.pose

    @property
    def id(self):
        return self['id']

    @property
    def res(self):
        return self['res']

    def copy(self):
        return ObjectState(self.id,
                           self.objclass,
                           self.pose,
                           res=self.res)

    @property
    def is_2d(self):
        return len(self.loc) == 2


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
                 camera_direction,
                 **kwargs):
        """
        Note: camera_direction is assumed to be None unless the robot is
        looking at a direction, in which case camera_direction is a string
        that indicates the direction e.g. look+x, or 'look'. It is _not_
        meant to be the pose of the camera; that is given by 'pose'.

        Note that robot pose is represented by a single tuple (x,y,th)
        or (x,y,z,qx,qy,qz,qw). This is distinct from pose representation
        in ObjectState as (position, orientation).
        """
        super().__init__("robot",
                         {"id": robot_id,
                          "pose": pose,
                          "objects_found": objects_found,
                          "camera_direction": camera_direction,
                          **kwargs})

    def __str__(self):
        return "{}({}, {})".format(type(self).__name__, self.pose, self.objects_found)

    def __hash__(self):
        return hash((self.id, self.pose))

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def objects_found(self):
        return self.attributes['objects_found']

    @property
    def camera_direction(self):
        return self.attributes['camera_direction']

    @property
    def id(self):
        return self.attributes['id']

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

    @property
    def rot(self):
        if self.is_2d:
            return self.pose[2]
        else:
            # 3d
            return self.pose[3:]

    def in_range(self, sensor, sobj, **kwargs):
        return sensor.in_range(sobj.loc, self["pose"], **kwargs)

    def loc_in_range(self, sensor, loc, **kwargs):
        return sensor.in_range(loc, self["pose"], **kwargs)


class RobotStateTopo(RobotState):
    def __init__(self,
                 robot_id,
                 pose,
                 objects_found,
                 camera_direction,
                 topo_nid,
                 topo_map_hashcode):
        super().__init__(robot_id,
                         pose,
                         objects_found,
                         camera_direction,
                         topo_nid=topo_nid,
                         topo_map_hashcode=topo_map_hashcode)

    @property
    def nid(self):
        return self['topo_nid']

    @property
    def topo_nid(self):
        return self['topo_nid']

    @property
    def topo_map_hashcode(self):
        return self['topo_map_hashcode']

    def __str__(self):
        return "{}({}, {}, nid={})".format(type(self).__name__, self.pose, self.objects_found, self.topo_nid)

    def __hash__(self):
        return hash((self.id, self.topo_nid, self.topo_map_hashcode))
