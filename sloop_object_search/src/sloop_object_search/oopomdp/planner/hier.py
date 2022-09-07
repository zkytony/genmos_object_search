import pomdp_py
from pomdp_py.utils import typ
from ..agent.topo3d import MosAgentTopo3D
from ..agent.topo2d import MosAgentTopo2D
from ..agent.common import MosAgent
from ..domain.action import StayAction

class HierPlanner(pomdp_py.Planner):
    def __init__(self, global_agent, **planner_params):
        self.planner_params = planner_params
        self._global_agent = global_agent
        global_planner_params = self.planner_params["global"]
        self._global_planner = pomdp_py.POUCT(**global_planner_params,
                                              rollout_policy=self._global_agent.policy_model)
        self._local_agent = None
        self._local_planner = None
        self.searching_locally = False

    @property
    def global_agent(self):
        return self._global_agent

    @property
    def local_agent(self):
        return self._local_agent

    def set_local_agent(self, local_agent):
        self._local_agent = local_agent

    def plan_local(self):
        if self._local_agent is None:
            raise RuntimeError("Local agent does not exist.")

        if self._local_planner is None:
            local_planner_params = self.planner_params["local"]
            self._local_planner = pomdp_py.POUCT(**local_planner_params,
                                                 rollout_policy=self._local_agent.policy_model)
        action = self._local_planner.plan(self._local_agent)
        action.robot_id = self._local_agent.robot_id
        return action

    def plan(self, agent):
        """
        The global planner always plans a subgoal. If the subgoal is
        "stay", then a local search agent is created, and action returned.
        """
        assert agent.robot_id == self._global_agent.robot_id,\
            "only plan for the agent given at construction."

        # plan with the global agent
        subgoal = self._global_planner.plan(self._global_agent)

        # DEBUGGING: JUST DO STAY
        topo_nid = agent.belief.b(agent.robot_id).mpe().topo_nid
        subgoal = StayAction(topo_nid)
        subgoal.robot_id = agent.robot_id
        print(typ.bold(typ.blue(f"Subgoal planned: {subgoal})")))

        if isinstance(subgoal, StayAction):
            self.searching_locally = True
            if self._local_agent is not None:
                # If the local agent is available, then plan with the local agent
                return self.plan_local()
            else:
                # This should trigger the creation of a local agent by the user of
                # this planner. Then, local search will be planned.
                return subgoal
        else:
            self.searching_locally = False
            return subgoal
