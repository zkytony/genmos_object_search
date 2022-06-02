cdef class GridMap:
    cdef public int width
    cdef public int length
    cdef public dict _obstacles
    cdef public dict _obstacle_states
    cdef public set _free_poses
    cdef public set _obstacle_poses
    cpdef valid_motions(self, int robot_id, tuple robot_pose, set all_motion_actions)
