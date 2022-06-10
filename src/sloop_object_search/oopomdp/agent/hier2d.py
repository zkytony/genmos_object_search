# Hierarchical agent, with a hierarchical planner
from sloop.agent import SloopAgent
from .topo2d import SloopMosTopo2DAgent
from sloop_object_search.oopomdp.action import MotionActionTopo, StayAction

import pomdp_py

class HierarchicalPlanner(pomdp_py.Planner):

    def __init__(self, planner_config):
        self._topo_agent = None
        self._subgoal_planner = None
        self._subgoal_handler = None
        self.planner_config = planner_config


    def plan(self, agent):
        if self._topo_agent is None:
            # First time plan
            assert isinstance(agent, SloopMosTopo2DAgent)
            self._topo_agent = agent
            self._subgoal_planner = pomdp_py.POUCT(**planner_config['subgoal_level'],
                                                   rollout_policy=agent.policy_model)

        if self._subgoal_handler is None:
            subgoal = self._subgoal_planner.plan(self._topo_agent)
            self._subgoal_handler = self.handle(subgoal)

        action = self._subgoal_handler.step()
        return action


    def handle(self, subgoal):
        if isinstance(subgoal, StayAction):
            return LocalSearchHandler(subgoal, self._topo_agent,
                                      self.planner_config['local_search'])
        elif isinstance(subgoal, MotionActionTopo):
            return NavTopoHandler(subgoal)

        elif isinstance(subgoal, FindAction):
            return FindHandler.create(goal, self)
