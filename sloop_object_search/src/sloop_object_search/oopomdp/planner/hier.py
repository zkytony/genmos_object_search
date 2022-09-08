import pomdp_py
from pomdp_py.utils import typ
from ..agent.topo3d import MosAgentTopo3D
from ..agent.topo2d import MosAgentTopo2D
from ..agent.common import MosAgent
from ..domain.action import StayAction
from ..models import belief
from sloop_object_search.utils.math import euclidean_dist

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

    def _verify_local_agent(self, local_agent):
        """The local agent is expected to have the same robot belief,
        as the global agent difference in pose due to noise."""
        if local_agent.robot_id != self.global_agent.robot_id + "_local":
            raise ValueError("Invalid robot id for local agent: {local_agent.robot_id}; "\
                             f"Expecting {self.global_agent.robot_id}_local")
        robot_state_local = local_agent.belief.b(local_agent.robot_id).mpe()
        robot_state_global = self.global_agent.belief.b(self.global_agent.robot_id).mpe()
        if robot_state_local.objects_found != robot_state_global.objects_found:
            raise ValueError("local and global agents have different belief in 'objects_found' "\
                             f"local: {robot_state_local.objects_found} "\
                             f"global: {robot_state_global.objects_found}")
        robot_pose_local_world = local_agent.search_region.to_world_pose(robot_state_local.pose)
        robot_pose_global_world = self.global_agent.search_region.to_world_pose(robot_state_global.pose)
        _pos_diff_tol = self.planner_params.get("local_global_pos_diff_tol", 3.0)
        assert euclidean_dist(robot_pose_local_world[:2], robot_pose_global_world[:2]) <= _pos_diff_tol,\
            "local and global robot poses differ too much."

    def set_local_agent(self, local_agent, init_object_beliefs_from_global=True):
        """If 'init_belief_from_global' is True, then will override
        the local_agent's object beliefs with that in the global agent, mapped
        to 3D. """
        self._verify_local_agent(local_agent)
        belief_conversion_params =\
            self.global_agent.agent_config["belief"].get("conversion", {})
        if init_object_beliefs_from_global:
            for objid in self.global_agent.belief.object_beliefs:
                if objid == self.global_agent.robot_id:
                    continue
                bobj2d = self.global_agent.belief.b(objid)
                bobj3d = belief.object_belief_2d_to_3d(
                    bobj2d, self.global_agent.search_region,
                    local_agent.search_region,
                    **belief_conversion_params)
                local_agent.belief.set_object_belief(objid, bobj3d)
        self._local_agent = local_agent

    def update_global_object_beliefs_from_local(self, normalizers_old):
        """"Assuming self.local_agent belief has been updated.
        Will update global agent's belief by projecting the
        3D belief of local agent down to 2D.

        normalizers_old: maps from objid to normalizer, indicating
        the normalizer of the octree belief before the most recent
        local agent belief update."""
        belief_conversion_params =\
            self.global_agent.agent_config["belief"].get("conversion", {})
        for objid in self.global_agent.belief.object_beliefs:
            if objid == self.global_agent.robot_id:
                continue
            bobj2d = self.global_agent.belief.b(objid)
            bobj3d = self.local_agent.belief.b(objid)
            bobj2d_updated = belief.update_2d_belief_by_3d(
                bobj2d, bobj3d, normalizers_old[objid], self.global_agent.search_region,
                self.local_agent.search_region,
                **belief_conversion_params)
            self.global_agent.belief.set_object_belief(objid, bobj2d_updated)

    def plan_local(self):
        if self._local_agent is None:
            raise RuntimeError("Local agent does not exist.")

        if self._local_planner is None:
            local_planner_params = self.planner_params["local"]
            self._local_planner = pomdp_py.POUCT(**local_planner_params,
                                                 rollout_policy=self._local_agent.policy_model)
        action = self._local_planner.plan(self._local_agent)
        action.robot_id = self._local_agent.robot_id
        action.name = action.name + "[local]"
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
