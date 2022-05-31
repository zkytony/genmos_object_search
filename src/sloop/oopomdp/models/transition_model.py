"""Defines the TransitionModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

Transition: deterministic
"""
import pomdp_py
from pomdp_py.framework.basics import State
import copy
from ..domain.state import *
from ..domain.observation import *
from ..domain.action import *

# Utility functions
def valid_pose(pose, width, length,
               state=None, check_collision=True,
               pose_objid=None):
    """
    Returns True if the given `pose` (x,y) is a valid pose;
    If `check_collision` is True, then the pose is only valid
    if it is not overlapping with any object pose in the environment state.
    """
    x, y = pose[:2]

    # Check collision with obstacles
    if check_collision and state is not None:
        object_poses = state.object_poses
        for objid in object_poses:
            if objid == pose_objid:
                continue
            if state.object_states[objid].objclass == "obstacle":
                if (x,y) == object_poses[objid]:
                    return False
    return in_boundary(pose, width, length)


def in_boundary(pose, width, length):
    # Check if in boundary
    x,y = pose[:2]
    if x >= 0 and x < width:
        if y >= 0 and y < length:
            if len(pose) == 3:
                th = pose[2]  # radian
                if th < 0 or th > 2*math.pi:
                    return False
            return True
    return False


####### Transition Model #######
class MosTransitionModel(pomdp_py.OOTransitionModel):
    """Object-oriented transition model; The transition model supports the
    multi-robot case, where each robot is equipped with a sensor; The
    multi-robot transition model should be used by the Environment, but
    not necessarily by each robot for planning.
    """
    def __init__(self,
                 dim, sensors, object_ids,
                 epsilon=1e-9,
                 no_look=False):
        """
        sensors (dict): robot_id -> Sensor
        for_env (bool): True if this is a robot transition model used by the
             Environment.  see RobotTransitionModel for details.
        """
        self._sensors = sensors
        transition_models = {objid: StaticObjectTransitionModel(objid, epsilon=epsilon)
                             for objid in object_ids
                             if objid not in sensors}
        for robot_id in sensors:
            transition_models[robot_id] = RobotTransitionModel(sensors[robot_id],
                                                               dim,
                                                               epsilon=epsilon,
                                                               no_look=no_look)
        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MosOOState(oostate.object_states)

class StaticObjectTransitionModel(pomdp_py.TransitionModel):
    """This model assumes the object is static."""
    def __init__(self, objid, epsilon=1e-9):
        self._objid = objid
        self._epsilon = epsilon

    def probability(self, next_object_state, state, action):
        if next_object_state != state.object_states[next_object_state['id']]:
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def sample(self, state, action):
        """Returns next_object_state"""
        return self.argmax(state, action)

    def argmax(self, state, action):
        """Returns the most likely next object_state"""
        return ObjectState(self._objid,
                           state.object_states[self._objid].objclass,
                           state.object_pose(self._objid))


class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""
    def __init__(self, sensor, dim, epsilon=1e-9, no_look=False):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self._sensor = sensor
        self._robot_id = sensor.robot_id
        self._dim = dim
        self._epsilon = epsilon
        self._no_look = no_look

    # useful for airsim demo
    MAPINFO = None
    MAPNAME = None
    FLIGHT_HEIGHT = 7
    LANDMARK_HEIGHTS = {}

    @classmethod
    def if_move_by(cls, robot_id, state, action, dim,
                   check_collision=True, robot_pose=None):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world.

        ## Useful for AirSim demo
        mapinfo (MapInfoDataset): If not None, checks
            whether robot next pose will be at a landmark and the flight height
            is below the height of that landmark.
        """
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        if state is None:
            assert robot_pose is not None, "Either provide state or provide robot pose"
        else:
            robot_pose = state.pose(robot_id)
        rx, ry, rth = robot_pose
        if action.scheme == "xy":
            dx, dy, th = action.motion
            rx += dx
            ry += dy
            rth = th
        elif action.scheme == "vw":
            # odometry motion model
            forward, angle = action.motion
            rth += angle  # angle (radian)
            rx = int(round(rx + forward*math.cos(rth)))
            ry = int(round(ry + forward*math.sin(rth)))
            rth = rth % (2*math.pi)

        valid = valid_pose((rx, ry, rth),
                           dim[0], dim[1],
                           state=state,
                           check_collision=check_collision,
                           pose_objid=robot_id)
        no_collide = True
        if cls.MAPINFO is not None:
            mapinfo = cls.MAPINFO
            map_name = cls.MAPNAME
            flight_height = cls.FLIGHT_HEIGHT
            landmark_heights = cls.LANDMARK_HEIGHTS

            lmk = mapinfo.landmark_at(map_name, (rx, ry))
            if lmk is not None:
                if lmk in mapinfo.streets(map_name):
                    lmk_height = landmark_heights.get(lmk, 0)
                else:
                    lmk_height = landmark_heights.get(lmk, 8)
                if flight_height < lmk_height:
                    no_collide = False
        if valid and no_collide:
            return (rx, ry, rth), not no_collide
        else:
            return robot_pose, not no_collide  # no change because change results in invalid pose

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return self._epsilon
        else:
            return 1.0 - self._epsilon

    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        if isinstance(state, RobotState):
            robot_state = state
        else:
            robot_state = state.object_states[self._robot_id]

        collision = robot_state["collision"]
        next_robot_pose = tuple(robot_state.pose)
        # camera direction is only not None when looking
        next_camera_direction = None
        next_objects_found = tuple(robot_state.objects_found)
        if isinstance(action, MotionAction):
            # motion action
            next_robot_pose, collision = \
                RobotTransitionModel.if_move_by(self._robot_id,
                                                state, action, self._dim)
            if self._no_look:
                # There's no Look action. So Motion action also leads to sensing.
                next_camera_direction = action.name
        elif isinstance(action, LookAction):
            if hasattr(action, "motion") and action.motion is not None:
                # rotate the robot
                next_robot_pose, collision = \
                    self._if_move_by(self._robot_id,
                                     state, action, self._dim)
            next_camera_direction = action.name
        elif isinstance(action, FindAction):
            robot_pose = state.pose(self._robot_id)
            z = self._sensor.observe(robot_pose, state)
            # Update "objects_found" set for target objects
            observed_target_objects = {objid
                                       for objid in z.objposes
                                       if (state.object_states[objid].objclass == "target"\
                                           and z.objposes[objid] != ObjectObservation.NULL)}
            next_objects_found = tuple(set(next_objects_found)\
                                       | set(observed_target_objects))
        return RobotState(self._robot_id,
                          next_robot_pose,
                          next_objects_found,
                          next_camera_direction,
                          collision=collision)

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
