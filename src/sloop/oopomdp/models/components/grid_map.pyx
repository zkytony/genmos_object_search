"""Optional grid map to assist collision avoidance during planning."""

from ..transition_model import RobotTransitionModel
from pomdp_py.framework.basics cimport State, Action
from ...domain.action import *
from ...domain.state import *

cdef class GridMap:
    """This map assists the agent to avoid planning invalid
    actions that will run into obstacles. Used if we assume
    the agent has a map. This map does not contain information
    about the object locations."""

    def __init__(self, width, length, obstacles):
        """
        Args:
            obstacles (dict): Map from objid to (x,y); The object is
                                   supposed to be an obstacle.
            width (int): width of the grid map
            length (int): length of the grid map
        """
        self.width = width
        self.length = length
        self._obstacles = obstacles
        # An MosOOState that only contains poses for obstacles;
        # This is to allow calling RobotTransitionModel.if_move_by
        # function.
        self._obstacle_states = {
            objid: ObjectState(objid, "obstacle", self._obstacles[objid])
            for objid in self._obstacles
        }
        # set of obstacle poses
        self._obstacle_poses = set({self._obstacles[objid]
                                    for objid in self._obstacles})
        # Free poses
        self._free_poses = set({
            (x,y)
            for x in range(self.width)
            for y in range(self.length)
            if (x,y) not in self._obstacle_poses
        })


    cpdef valid_motions(self, int robot_id, tuple robot_pose, set all_motion_actions):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        cdef State state
        cdef Action motion_action
        cdef tuple next_pose
        cdef set valid

        state = MosOOState(self._obstacle_states)
        state.set_object_state(robot_id,
                               RobotState(robot_id, robot_pose, None, None))

        valid = set({})
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_pose = RobotTransitionModel.if_move_by(robot_id, state,
                                                        motion_action, (self.width, self.length))[0]
            if next_pose != robot_pose:
                # robot moved --> valid motion
                valid.add(motion_action)
        return valid

    @property
    def free_poses(self):
        return self._free_poses

    @property
    def obstacle_poses(self):
        return self._obstacle_poses

    def has_obstacle(self, x, y):
        return (x,y) in self._obstacle_poses

    def within_range(self, x, y):
        return (0 <= x and x < self.width)\
            and (0 <= y and y < self.length)

    def get_neighbors(self, pose, all_motion_actions, include_angle=False):
        neighbors = {}
        if len(pose) == 2:
            pose = (pose[0], pose[1], 0)
        for motion_action in all_motion_actions:
            if not isinstance(motion_action, MotionAction):
                raise ValueError("This (%s) is not a motion action" % str(motion_action))

            next_pose = RobotTransitionModel.if_move_by(None, None,
                                                        motion_action,
                                                        (self.width, self.length),
                                                        check_collision=False,
                                                        robot_pose=pose)[0]
            if next_pose[:2] not in self._obstacle_poses:
                if include_angle:
                    # Use (x,y,th) as key
                    neighbors[next_pose] = motion_action
                else:
                    # Only use (x,y) as key
                    neighbors[next_pose[:2]] = motion_action
        return neighbors
