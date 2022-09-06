import pomdp_py
from pomdp_py.utils import typ
from .topo3d import MosAgentTopo3D
from .topo2d import MosAgentTopo2D
from .common import MosAgent
from ..domain.action import StayAction

class HierMosAgent(MosAgentTopo2D):
    """a hierarchical search agent holds a global search agent and possibly a local
    search agent (if it is searching locally). The hierarchical search agent is
    created given a 2D search region. If it is searching locally, then a 3D
    search region can be provided through the 'create_local_agent' method which
    builds a local search agent.
    """
    def __init__(self, agent_config, search_region, init_robot_pose_dist,
                 init_object_beliefs=None):
        super().__init__(agent_config, search_region, init_robot_pose_dist,
                         init_object_beliefs=init_object_beliefs)
        self._local_agent = None
        self.searching_locally = True    # The agent has started to search locally

    def set_local_agent(self, local_agent):
        self._local_agent = local_agent

    @property
    def local_agent(self):
        return self._local_agent

    def create_local_agent(self, robot_loc, local_search_region):
        """Create a local agent with initial belief based on the global agent's belief.
        Args:
            robot_loc (RobotLocalization): 3D robot localization, in local region frame
            local_search_region (SearchRegion3D): 3D search regionx
        """
        if self._local_agent is not None:
            local_agent_config = self.make_local_agent_config()
            self._local_agent = MosAgentTopo3D(local_agent_config,
                                               local_search_region,
                                               robot_loc)



class HierPlanner(pomdp_py.Planner):
    def __init__(self, hier_agent, **planner_params):
        if not isinstance(hier_agent, HierMosAgent):
            raise TypeError("HierarchicalPlanner requires global search "\
                            "agent to be of type MosAgentTopo2D, but got "\
                            f"{type(topo_agent)}")

        self.planner_params = planner_params
        self._hier_agent = hier_agent
        global_planner_params = self.planner_params["global"]
        self._global_planner = pomdp_py.POUCT(**global_planner_params,
                                              rollout_policy=self._hier_agent.policy_model)
        self._local_planner = None

    def plan_local(self):
        if self._local_planner is None:
            local_planner_params = self.planner_params["local"]
            self._local_planner = pomdp_py.POUCT(**local_planner_params,
                                                 rollout_policy=self._hier_agent.local_agent.policy_model)
        action = self._local_planner.plan(self._hier_agent.local_agent)
        return action

    def plan(self, agent):
        """
        The global planner always plans a subgoal. If the subgoal is
        "stay", then a local search agent is created, and action returned.
        """
        assert agent.robot_id == self._hier_agent.robot_id,\
            "only plan for the agent given at construction."

        # plan with the global agent
        subgoal = self._global_planner.plan(self._hier_agent)

        # DEBUGGING: JUST DO STAY
        topo_nid = agent.belief.b(agent.robot_id).mpe().topo_nid
        subgoal = StayAction(topo_nid)
        print(typ.bold(typ.blue(f"Subgoal planned: {subgoal})")))

        if isinstance(subgoal, StayAction):
            self._hier_agent.searching_locally = True  # this is a server-set attribute
            if self._hier_agent.local_agent is not None:
                # If the local agent is available, then plan with the local agent
                return self.plan_local()
            else:
                # This should trigger the creation of a local agent by the user of
                # this planner. Then, local search will be planned.
                return subgoal
        else:
            return subgoal
