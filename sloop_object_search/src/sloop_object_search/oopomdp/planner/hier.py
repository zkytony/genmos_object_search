# import pomdp_py
# from pomdp_py.utils import typ
# from ..agent.topo2d import MosAgentTopo2D

# class HierarchicalPlanner(pomdp_py.Planner):
#     """
#     Hierarchical planner. Interacts with a global search agent,
#     and manages a local search agent, if necessary. The global
#     search agent is a 2D topo agent (topo2d.py), while the local search agent
#     is a 3D topo agent (topo3d.py).
#     """
#     def __init__(self, topo2d_agent, **planner_params):
#         if not isinstance(topo2d_agent, MosAgentTopo2D):
#             raise TypeError("HierarchicalPlanner requires global search "\
#                             "agent to be of type MosAgentTopo2D, but got "\
#                             f"{type(topo_agent)}")

#         self.planner_params = planner_params
#         self._global_agent = topo2d_agent
#         global_planner_params = self.planner_params["global"]
#         self._global_planner = pomdp_py.POUCT(**global_planner_params,
#                                               rollout_policy=self._topo_agent.policy_model)
#         self._local_agent = None
#         self._local_planner = None

#     def plan(self, agent):
#         """
#         The global planner always plans a subgoal. If the subgoal is
#         "stay", then a local search agent is created, and action returned.
#         """
#         assert agent.robot_id == self._topo_agent.robot_id,\
#             "only plan for the agent given at construction."
#         subgoal = self._global_planner.plan(self._topo_agent)
#         print(typ.bold(typ.blue(f"Subgoal planned: {subgoal})")))

#         if isinstance(subgoal, StayAction):
#             # create a local search agent. Unless one exists already
#             if self._local_agent is not None:
#                 assert self._local_planner is not None
#                 return self._local_planner.plan(self._local_agent)
#             else:
#                 self._local_agent = create_local_search_agent(self._global_agent)


#         if self._current_subgoal is None:
#             self._current_subgoal = subgoal




#         if self._subgoal_handler is None:
#             self._subgoal_handler = self.handle(subgoal)
#         else:
#             if subgoal != self._subgoal_handler.subgoal:
#                 self._subgoal_handler = self.handle(subgoal)
#         action = self._subgoal_handler.step()
#         return action

#     def handle(self, subgoal):
#         if isinstance(subgoal, StayAction):
#             return LocalSearchHandler(subgoal, self._topo_agent, self._mos2d_agent,
#                                       self.planner_config['local_search'])
#         elif isinstance(subgoal, MotionActionTopo):
#             # we don't handle navigation. Just


#             # set a destination pose; the orientation is determined by the transition
#             # model given a random state sampled from belief.
#             rnd_state = self._topo_agent.belief.random()
#             robot_trans_model = self._topo_agent.transition_model[self._topo_agent.robot_id]
#             subgoal.dst_pose = robot_trans_model.sample(rnd_state, subgoal).pose

#             # Check whether we want to plan individual movements to fulfill the navigation
#             # subgoal, or if we just want to directly output the navigation subgoal
#             plan_nav_actions = self.planner_config.get("plan_nav_actions", True)
#             if plan_nav_actions:
#                 return NavTopoHandler(subgoal, self._topo_agent, self._mos2d_agent)
#             else:
#                 return NavTopoIdentityHandler(subgoal, self._topo_agent)

#         elif isinstance(subgoal, FindAction):
#             return FindHandler(subgoal)

#     def update(self, agent, action, observation):
#         if self._subgoal_handler is not None:
#             self._subgoal_handler.update(action, observation)
#             self._subgoal_planner.update(agent, action, observation)

#             if self._subgoal_handler.done:
#                 self._subgoal_handler = None
#                 self._current_subgoal = None
