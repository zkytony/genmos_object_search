import pomdp_py
from ..domain.action import FindAction
from ..domain.state import RobotState2D
from ..models.belief import BeliefBasic2D
from sloop_object_search.utils.misc import import_class

class SubgoalHandler:
    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError

    def _copy_topo_agent_belief(self):
        robot_id = self._topo_agent.robot_id
        srobot_topo = self._topo_agent.belief.mpe().s(robot_id)
        srobot = RobotState2D(robot_id,
                              srobot_topo.pose,
                              srobot_topo.objects_found,
                              srobot_topo.camera_direction)
        belief = BeliefBasic2D(srobot,
                               self._topo_agent.target_objects,
                               object_beliefs=dict(self._topo_agent.belief.object_beliefs))
        self._mos2d_agent.set_belief(belief)


class LocalSearchHandler(SubgoalHandler):
    def __init__(self,
                 subgoal,
                 topo_agent,
                 mos2d_agent,
                 local_search_planner_args):
        self._topo_agent = topo_agent  # parent
        self._mos2d_agent = mos2d_agent
        self._copy_topo_agent_belief()

        planner_class = local_search_planner_args["planner"]
        self.planner = import_class(planner_class)(
            **local_search_planner_args.get("planner_args", {}),
            rollout_policy=self._mos2d_agent.policy_model
        )

    def step(self):
        return self.planner.plan(self.agent)

    def update(self, action, observation):
        self.planner.update(self.agent, action, observation)
        self.agent.set_belief(self._topo_agent.belief)
        srobot_topo = self._topo_agent.belief.mpe().s(topo_agent.robot_id)
        srobot = RobotState2D(topo_agent.robot_id,
                              srobot_topo.pose,
                              srobot_topo.objects_found,
                              srobot_topo.camera_direction)
        self.agent.belief.set_object_belief(self.agent.robot_id,
                                            pomdp_py.Histogram({srobot: 1.0}))


class FindHandler(SubgoalHandler):
    def __init__(self, subgoal):
        pass

    def step(self):
        return FindAction()

    def update(self):
        pass


class NavTopoHandler(SubgoalHandler):
    def __init__(self, subgoal, topo_agent, mos2d_agent):
        self._topo_agent = topo_agent
        self._mos2d_agent = mos2d_agent
        self._copy_topo_agent_belief()
        srobot_topo = topo_agent.belief.mpe().s(topo_agent.robot_id)
        assert subgoal.src_nid == srobot_topo.topo_nid

        # Compute navigation plan
        movements_dict = self._mos2d_agent.policy_model.movements
        navigation_actions = {(action_name, movements_dict[action_name].motion)
                              for action_name in movements_dict}
        robot_trans_model = self._mos2d_agent.transition_model[self._mos2d_agent.robot_id]
        self._nav_plan = find_navigation_plan(src_pose, dst_pose,
                                              navigation_actions,
                                              robot_trans_model.reachable_positions,
                                              diagonal_ok=True,
                                              return_pose=True)
        self._index = 0

    def step(self):
        return self._nav_plan[self._index]

    def update(self):
        self._index += 1


##### Auxiliary functions for navigation handler ###############
def find_navigation_plan(start, goal, navigation_actions,
                         reachable_positions,
                         goal_distance=0.0,
                         grid_size=None,
                         angle_tolerance=5,
                         diagonal_ok=False,
                         return_pose=False,
                         debug=False):
    """
    FUNCTION ORIGINALLY FROM thortils

    Returns a navigation plan as a list of navigation actions. Uses A*

    Recap of A*: A* selects the path that minimizes

    f(n)=g(n)+h(n)

    where n is the next node on the path, g(n) is the cost of the path from the
    start node to n, and h(n) is a heuristic function that estimates the cost of
    the cheapest path from n to the goal.  If the heuristic function is
    admissible, meaning that it never overestimates the actual cost to get to
    the goal, A* is guaranteed to return a least-cost path from start to goal.

    Args:
        start (tuple): position, rotation of the start
        goal (tuple): position, rotation of the goal n
        navigation_actions (list): list of navigation actions,
            represented as ("ActionName", (forward, h_angles, v_angles)),
        goal_distance (bool): acceptable minimum euclidean distance to the goal
        grid_size (float): size of the grid, typically 0.25. Only
            necessary if `diagonal_ok` is True
        diagonal_ok (bool): True if 'MoveAhead' can move
            the robot diagonally.
        return_pose (bool): True if return a list of {"action": <action>, "next_pose": <pose>} dicts
        debug (bool): If true, returns the expanded poses
    Returns:
        a list consisting of elements in `navigation_actions`
    """
    if type(reachable_positions) != set:
        reachable_positions = set(reachable_positions)

    # Map angles in start and goal to be within 0 to 360 (see top comments)
    start_rotation = normalize_angles(start[1])
    goal_rotation = normalize_angles(goal[1])
    start = (start[0], start_rotation)
    goal = (goal[0], goal_rotation)

    # The priority queue
    worklist = PriorityQueue()
    worklist.push(start, _nav_heuristic(start, goal))

    # cost[n] is the cost of the cheapest path from start to n currently known
    cost = {}
    cost[start] = 0

    # comefrom[n] is the node immediately preceding node n on the cheapeast path
    comefrom = {}

    # keep track of visited poses
    visited = set()

    if debug:
        _expanded_poses = []

    while not worklist.isEmpty():
        current_pose = worklist.pop()
        if debug:
            _expanded_poses.append(current_pose)
        if _round_pose(current_pose) in visited:
            continue
        if _same_pose(current_pose, goal,
                      tolerance=goal_distance,
                      angle_tolerance=angle_tolerance):
            if debug:
                plan = _reconstruct_plan(comefrom,
                                         current_pose,
                                         return_pose=True)
                return plan, _expanded_poses
            else:
                return _reconstruct_plan(comefrom, current_pose,
                                         return_pose=return_pose)

        for action in navigation_actions:
            next_pose = transform_pose(current_pose, action,
                                       grid_size=grid_size,
                                       diagonal_ok=diagonal_ok)
            if not _valid_pose(_round_pose(next_pose), reachable_positions):
                continue

            new_cost = cost[current_pose] + _cost(action)
            if new_cost < cost.get(next_pose, float("inf")):
                cost[next_pose] = new_cost
                worklist.push(next_pose, cost[next_pose] + _nav_heuristic(next_pose, goal))
                comefrom[next_pose] = (current_pose, action)

        visited.add(current_pose)

    # no path found
    if debug:
        return None, _expanded_poses
    else:
        return None


def transform_pose(robot_pose, action, schema="vw",
                   diagonal_ok=False, grid_size=None):
    """Transform pose of robot in 2D;
    This is a generic function, not specific to Thor.

    Args:
       robot_pose (tuple): Either 2d pose (x,y,yaw,pitch), or (x,y,yaw).
              or a tuple (position, rotation):
                  position (tuple): tuple (x, y, z)
                  rotation (tuple): tuple (x, y, z); pitch, yaw, roll.
       action:
              ("ActionName", delta), where delta is the change, format dependent on schema

       grid_size (float or None): If None, then will not
           snap the transformed x,y to grid.

       diagonal_ok (bool): True if it is ok to go diagonally,
           even though the traversed distance is longer than grid_size.

    Returns the transformed pose in the same form as input
    """
    action_name, delta = action
    if schema == "vw":
        x, z, pitch, yaw = _simplify_pose(robot_pose)
        new_pose = _move_by_vw((x, z, pitch, yaw), delta,
                               grid_size=grid_size, diagonal_ok=diagonal_ok)
    elif schema == "vw2d":
        x, z, yaw = robot_pose
        new_pose = _move_by_vw2d((x, z, yaw), delta,
                                 grid_size=grid_size, diagonal_ok=diagonal_ok)
    else:
        raise ValueError("Unknown schema")

    if _is_full_pose(robot_pose):
        new_rx, new_rz, new_yaw, new_pitch = new_pose
        return (new_rx, robot_pose[0][1], new_rz),\
            (new_pitch, new_yaw, robot_pose[1][2])
    else:
        return new_pose
