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

##################### 2D Motion Action ##############################
MOTION_SCHEME="vw"  # can be either xy or vw
STEP_SIZE=3
class MotionAction2D(MotionAction):
    # scheme 1 (vx,vy,th)
    EAST = (STEP_SIZE, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST = (-STEP_SIZE, 0, math.pi)
    NORTH = (0, -STEP_SIZE, 3*math.pi/2)
    SOUTH = (0, STEP_SIZE, math.pi/2)
    # scheme 2 (vt, vw) translational, rotational velocities.
    FORWARD = (STEP_SIZE, 0)
    BACKWARD = (-STEP_SIZE, 0)

    LEFT_45 = (0, -45.0)  # left 45 deg
    RIGHT_45 = (0, 45.0)  # right 45 deg

    LEFT_90 = (0, -90.0)  # left 90 deg
    RIGHT_90 = (0, 90.0)  # right 90 deg

    LEFT = LEFT_45
    RIGHT = RIGHT_45

    def __init__(self, motion, distance_cost=1,
                 motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        self.motion = motion
        self.distance_cost = distance_cost
        if motion_name is None:
            motion_name = str(motion)
        super().__init__("move-{}({})".format(motion_name, motion))

    @property
    def dyaw(self):
        return self.motion[1]

# Define some constant actions
MoveEast = MotionAction2D(MotionAction2D.EAST, motion_name="East")
MoveWest = MotionAction2D(MotionAction2D.WEST, motion_name="West")
MoveNorth = MotionAction2D(MotionAction2D.NORTH, motion_name="North")
MoveSouth = MotionAction2D(MotionAction2D.SOUTH, motion_name="South")
MoveForward = MotionAction2D(MotionAction2D.FORWARD, motion_name="Forward")
MoveBackward = MotionAction2D(MotionAction2D.BACKWARD, motion_name="Backward")
MoveLeft = MotionAction2D(MotionAction2D.LEFT, motion_name="TurnLeft")
MoveRight = MotionAction2D(MotionAction2D.RIGHT, motion_name="TurnRight")

Look = LookAction()
Find = FindAction()

if MOTION_SCHEME == "xy":
    ALL_MOTION_ACTIONS = {MoveEast, MoveWest, MoveNorth, MoveSouth}
elif MOTION_SCHEME == "vw":
    ALL_MOTION_ACTIONS = {MoveForward, MoveLeft, MoveRight}
else:
    raise ValueError("motion scheme '%s' is invalid" % MOTION_SCHEME)

NAME_TO_ACTION = {}
for action in ALL_MOTION_ACTIONS | {Look, Find}:
    NAME_TO_ACTION[action.name] = action
