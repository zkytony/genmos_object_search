"""
SLOOP action. Here, we provide a few generic
action types, useful for multi-object search
However, no specific implementation is provided, as that
is the job of individual domains
"""
import math
import pomdp_py

##################### Generic definitions ###########################
class MotionAction(pomdp_py.SimpleAction):
    """MotionAction moves the robot.
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
        super().__init__("move-{}".format(motion_name))

    @property
    def dyaw(self):
        return self.motion[1]

def basic_discrete_moves2d(step_size=1, h_rotation=45.0, back=False):
    """returns mapping from action name to Action"""
    # scheme vw: (vt, vw) translational, rotational velocities.
    FORWARD = (step_size, 0)
    BACKWARD = (-step_size, 0)
    LEFT = (0, -h_rotation)  # left 45 deg
    RIGHT = (0, h_rotation)  # right 45 deg
    MoveForward = MotionAction2D(FORWARD, motion_name="Forward")
    MoveBackward = MotionAction2D(BACKWARD, motion_name="Backward")
    TurnLeft = MotionAction2D(LEFT, motion_name="TurnLeft")
    TurnRight = MotionAction2D(RIGHT, motion_name="TurnRight")
    if back:
        return [MoveForward, MoveBackward, TurnLeft, TurnRight]
    else:
        return [MoveForward, TurnLeft, TurnRight]

################## Topological movement ###########################
class MotionActionTopo(MotionAction):
    def __init__(self, src_nid, dst_nid, gdist=None,
                 cost_scaling_factor=1.0, atype="move"):
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.gdist = gdist
        self._cost_scaling_factor = cost_scaling_factor
        super().__init__("{}({}->{})".format(atype, self.src_nid, self.dst_nid))

    @property
    def step_cost(self):
        return min(-(self.gdist * self._cost_scaling_factor), -1)

class StayAction(MotionActionTopo):
    def __init__(self, nid, cost_scaling_factor=1.0):
        super().__init__(nid, nid, gdist=0.0, cost_scaling_factor=1.0, atype="stay")
