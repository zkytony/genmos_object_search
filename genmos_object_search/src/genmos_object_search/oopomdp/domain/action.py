"""
GenMOS action. Here, we provide a few generic
action types, useful for multi-object search
However, no specific implementation is provided, as that
is the job of individual domains
"""
import math
import pomdp_py
from genmos_object_search.utils.math import euler_to_quat

##################### Generic definitions ###########################
class MotionAction(pomdp_py.SimpleAction):
    """MotionAction moves the robot relative to its current pose.
    The specific definition is domain-dependent"""
    def __repr__(self):
        return str(self)

class FindAction(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("find")
    def __repr__(self):
        return str(self)

class LookAction(pomdp_py.SimpleAction):
    def __init__(self):
        super().__init__("look")

# Define some constant actions
Look = LookAction()
Find = FindAction()

##################### 2D Motion Action ##############################
MOTION_SCHEME="vw"  # can be either xy or vw
STEP_SIZE=3
class MotionAction2D(MotionAction):
    def __init__(self, motion, step_cost=-1,
                 motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        self.motion = motion
        self.step_cost = step_cost
        if motion_name is None:
            motion_name = str(motion)
        self.motion_name = motion_name
        super().__init__("move2d-{}".format(motion_name))

    @property
    def dyaw(self):
        return self.motion[1]

def basic_discrete_moves2d(step_size=1, h_rotation=45.0, back=False, yaxis="down"):
    """returns mapping from action name to Action"""
    # scheme vw: (vt, vw) translational, rotational velocities.
    FORWARD = (step_size, 0)
    BACKWARD = (-step_size, 0)
    if yaxis == "down":
        LEFT = (0, -h_rotation)  # left 45 deg
        RIGHT = (0, h_rotation)  # right 45 deg
    elif yaxis == "up":
        LEFT = (0, h_rotation)  # left 45 deg
        RIGHT = (0, -h_rotation)  # right 45 deg
    else:
        raise ValueError(f"invalid yaxis {yaxis}")
    MoveForward = MotionAction2D(FORWARD, motion_name="Forward")
    MoveBackward = MotionAction2D(BACKWARD, motion_name="Backward")
    TurnLeft = MotionAction2D(LEFT, motion_name="TurnLeft")
    TurnRight = MotionAction2D(RIGHT, motion_name="TurnRight")
    if back:
        return [MoveForward, MoveBackward, TurnLeft, TurnRight]
    else:
        return [MoveForward, TurnLeft, TurnRight]

##################### 3D Motion Action ##############################
class MotionAction3D(MotionAction):
    """The motion tuple for 3D is (dx, dy, dz), (dthx, dthy, dthz)
    where dthx, dthy, dthz are rotations with respect to x, y, z
    axes"""
    def __init__(self, motion, step_cost=-1,
                 motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        self.motion = motion
        self.step_cost = step_cost
        if motion_name is None:
            motion_name = str(motion)
        self.motion_name = motion_name
        super().__init__("move3d-{}".format(motion_name))

def basic_discrete_moves3d(step_size=1, rotation=90.0, scheme="axis"):
    """returns mapping from action name to Action"""
    # This comes from the AXIS motion model in mos3d
    schemes = {
        "axis": {
            "-x": ((-step_size,0,0), (0,0,0)),
            "+x": ((step_size,0,0),  (0,0,0)),
            "-y": ((0,-step_size,0), (0,0,0)),
            "+y": ((0,step_size,0),  (0,0,0)),
            "-z": ((0,0,-step_size), (0,0,0)),
            "+z": ((0,0,step_size),  (0,0,0)),
            "+thx": ((0,0,0), (rotation,0,0)),
            "-thx": ((0,0,0), (-rotation,0,0)),
            "+thy": ((0,0,0), (0,rotation,0)),
            "-thy": ((0,0,0), (0,-rotation,0)),
            "+thz": ((0,0,0), (0,0,rotation)),
            "-thz": ((0,0,0), (0,0,-rotation))
        },
        "forward": {
            "forward":  ( step_size, (0,0,0)  ),
            "backward": (-step_size, (0,0,0)  ),
            "+thx":     ( 0, (rotation,0,0) ),
            "-thx":     ( 0, (-rotation,0,0)),
            "+thy":     ( 0, (0,rotation,0) ),
            "-thy":     ( 0, (0,-rotation,0)),
            "+thz":     ( 0, (0,0,rotation) ),
            "-thz":     ( 0, (0,0,-rotation))
        }
    }

    actions = []
    for name in schemes[scheme]:
        motion = schemes[scheme][name]
        action = MotionAction3D(motion, motion_name=f"{scheme}({name})")
        actions.append(action)
    return actions


################## Topological movement (General for 2D/3D) ###########################
class MotionActionTopo(MotionAction):
    def __init__(self, src_nid, dst_nid, topo_map_hashcode=None, distance=None,
                 cost_scaling_factor=1.0, atype="move"):
        """distance: distance between source and destination nodes.
                     (ideally, geodesic distance)"""
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.distance = distance
        self.topo_map_hashcode = topo_map_hashcode
        self._cost_scaling_factor = cost_scaling_factor

        if topo_map_hashcode is not None:
            action_name = "{}({}->{}@{})".format(atype, self.src_nid, self.dst_nid, self.topo_map_hashcode[:4])
        else:
            action_name = "{}({}->{})".format(atype, self.src_nid, self.dst_nid)
        super().__init__(action_name)

    @property
    def step_cost(self):
        return min(-(self.distance * self._cost_scaling_factor), -1)

class StayAction(MotionActionTopo):
    def __init__(self, nid, topo_map_hashcode=None, cost_scaling_factor=1.0):
        super().__init__(nid, nid, topo_map_hashcode=topo_map_hashcode,
                         distance=0.0, cost_scaling_factor=1.0, atype="stay")
