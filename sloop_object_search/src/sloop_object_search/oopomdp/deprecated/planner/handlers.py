from collections import deque
import pomdp_py
from ...domain.action import FindAction
from ...domain.state import RobotState
from ..domain.state import RobotStateTopo # deprecated
from ..models.transition_model import RobotTransBasic2D
from ..models.belief import BeliefBasic2D
from genmos_object_search.utils.misc import import_class
from genmos_object_search.utils.math import normalize_angles, euclidean_dist, fround
from genmos_object_search.utils.algo import PriorityQueue

class SubgoalHandler:
    @classmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    def _copy_topo_agent_belief(self):
        robot_id = self._topo_agent.robot_id
        srobot_topo = self._topo_agent.belief.mpe().s(robot_id)
        srobot = RobotState(robot_id,
                            srobot_topo.pose,
                            srobot_topo.objects_found,
                            srobot_topo.camera_direction)
        belief = BeliefBasic2D(self._topo_agent.target_objects,
                               robot_state=srobot,
                               object_beliefs=dict(self._topo_agent.belief.object_beliefs))
        self._mos2d_agent.set_belief(belief)


class LocalSearchHandler(SubgoalHandler):
    def __init__(self,
                 subgoal,
                 topo_agent,
                 mos2d_agent,
                 local_search_planner_args):
        self.subgoal = subgoal
        self._topo_agent = topo_agent  # parent
        self._mos2d_agent = mos2d_agent
        self._copy_topo_agent_belief()

        planner_class = local_search_planner_args["planner"]
        self.planner = import_class(planner_class)(
            **local_search_planner_args.get("planner_args", {}),
            rollout_policy=self._mos2d_agent.policy_model
        )

    def step(self):
        action = self.planner.plan(self._mos2d_agent)
        if hasattr(self._mos2d_agent, "tree") and self._mos2d_agent.tree is not None:
            _dd = pomdp_py.utils.TreeDebugger(self._mos2d_agent.tree)
            print("vvvvvvvvvvvvvvvvvvv local planner vvvvvvvvvvvvvvvvvvvv-")
            _dd.p(1)
            print("^^^^^^^^^^^^^^^^^^^ local planner ^^^^^^^^^^^^^^^^^^^^-")
        return action

    def update(self, action, observation):
        self.planner.update(self._mos2d_agent, action, observation)
        self._copy_topo_agent_belief()

    @property
    def done(self):
        return False


class FindHandler(SubgoalHandler):
    def __init__(self, subgoal):
        self.subgoal = subgoal

    def step(self):
        return FindAction()

    def update(self, *args):
        pass

    @property
    def done(self):
        return True


class NavTopoIdentityHandler(SubgoalHandler):
    """Just returns the subgoal as action; but performs
    update appropriately for navigation topo subgoal"""
    def __init__(self, subgoal, topo_agent):
        self.subgoal = subgoal
        self._topo_agent = topo_agent
        self._done = False

    def step(self):
        return self.subgoal

    @property
    def done(self):
        return self._done

    def update(self, action, observation):
        self._done = True
        zrobot = observation.z(self._topo_agent.robot_id)
        topo_nid = self._topo_agent.topo_map.closest_node(*zrobot.pose[:2])
        srobot_topo = RobotStateTopo(zrobot.robot_id,
                                     zrobot.pose,
                                     zrobot.objects_found,
                                     zrobot.camera_direction,
                                     topo_nid)
        self._topo_agent.belief.set_object_belief(self._topo_agent.robot_id,
                                                  pomdp_py.Histogram({srobot_topo: 1.0}))


class NavTopoHandler(SubgoalHandler):
    def __init__(self, subgoal, topo_agent, mos2d_agent):
        self.subgoal = subgoal
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
        start = (srobot_topo.pose[:2], (srobot_topo.pose[2],))
        goal = (subgoal.dst_pose[:2], (subgoal.dst_pose[2],))
        self._nav_plan = find_navigation_plan(start, goal,
                                              navigation_actions,
                                              robot_trans_model.reachable_positions,
                                              return_pose=True)
        self._index = 0
        self._done = False

    @property
    def done(self):
        return self._done

    def step(self):
        if not self._done:
            action_name = self._nav_plan[self._index]['action'][0]
            return self._mos2d_agent.policy_model.movements[action_name]

    def update(self, action, observation):
        if self._index < len(self._nav_plan) - 1:
            self._index += 1
        else:
            self._done = True
            zrobot = observation.z(self._topo_agent.robot_id)
            topo_nid = self._topo_agent.topo_map.closest_node(*zrobot.pose[:2])
            srobot_topo = RobotStateTopo(zrobot.robot_id,
                                         zrobot.pose,
                                         zrobot.objects_found,
                                         zrobot.camera_direction,
                                         topo_nid)
            self._topo_agent.belief.set_object_belief(self._topo_agent.robot_id,
                                                      pomdp_py.Histogram({srobot_topo: 1.0}))


##### Auxiliary functions for navigation handler ###############
def _nav_heuristic(pose, goal):
    """Returns underestimate of the cost from pose to goal
    pose tuple(position, rotation); goal tuple(position, rotation)"""
    return euclidean_dist(pose[0], goal[0])

def _reconstruct_plan(comefrom, end_node, return_pose=False):
    """Returns the plan from start to end_node; The dictionary `comefrom` maps from node
    to parent node and the edge (i.e. action)."""
    plan = deque([])
    node = end_node
    while node in comefrom:
        parent_node, action = comefrom[node]
        if return_pose:
            plan.appendleft({"action": action, "next_pose": node})
        else:
            plan.appendleft(action)
        node = parent_node
    return list(plan)

def _cost(action):
    """
    action is (movement_str, (forward, h_angle, v_angle))
    """
    forward, h_angle = action[1]
    cost = 0
    if forward != 0:
        cost += 1
    if h_angle != 0:
        cost += 1
    return cost

def _round_pose(full_pose):
    return (fround('int', full_pose[0]),
            fround('int', full_pose[1]))

def _same_pose(pose1, pose2, tolerance=1e-4, angle_tolerance=5):
    """
    Returns true if pose1 and pose2 are of the same pose;

    Args:
       tolerance (float): Euclidean distance tolerance
       angle_tolerance (float): Angular tolerance;
          Instead of relying on this tolerance, you
          should make sure the goal pose's rotation
          can be achieved exactly by taking the
          rotation actions.
    """
    x1, y1 = pose1[0]
    th1 = pose1[1][0]

    x2, y2 = pose2[0]
    th2 = pose2[1][0]

    return euclidean_dist((x1, y1), (x2, y2)) <= tolerance\
        and abs(th1 - th2) <= angle_tolerance


def find_navigation_plan(start, goal, navigation_actions,
                         reachable_positions,
                         goal_distance=0.0,
                         angle_tolerance=5,
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
            next_pose = transform_pose(current_pose, action)
            if not _round_pose(next_pose)[0] in reachable_positions:
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


def transform_pose(robot_pose, action):
    """Transform pose of robot in 2D;
    This is a generic function, not specific to Thor.

    Args:
       robot_pose (tuple): Either 2d pose (x,y,yaw,pitch), or (x,y,yaw).
              or a tuple (position, rotation):
                  position (tuple): tuple (x, y, z)
                  rotation (tuple): tuple (x, y, z); pitch, yaw, roll.
       action:
              ("ActionName", delta), where delta is the change, format dependent on schema

    Returns the transformed pose in the same form as input
    """
    action_name, delta = action
    x, y = robot_pose[0]
    yaw = robot_pose[1][0]
    nx, ny, nyaw = RobotTransBasic2D.transform_pose((x, y, yaw), delta)
    return ((nx, ny), (nyaw,))
