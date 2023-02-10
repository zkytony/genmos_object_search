# Hierarchical agent, with a hierarchical planner
from pomdp_py.utils import typ
from sloop.agent import SloopAgent
from ..agent.topo2d import SloopMosTopo2DAgent
from ..agent.basic2d import MosBasic2DAgent
from ..models.belief import BeliefBasic2D
from ...domain.state import RobotState
from ...domain.action import MotionActionTopo, StayAction, FindAction
from .handlers import (LocalSearchHandler,
                       NavTopoHandler,
                       FindHandler,
                       NavTopoIdentityHandler)

import pomdp_py

class HierarchicalPlanner(pomdp_py.Planner):

    """
    configuration example (YAML format):

       planner_config:
         planner: genmos_object_search.oopomdp.planner.hier2d.HierarchicalPlanner
         high_level_planner_args:
           exploration_const: 1000
           max_depth: 20
           planning_time: 0.25
         local_search:
           planner: pomdp_py.POUCT
           planner_args:
             exploration_const: 1000
             max_depth: 10
             planning_time: 0.15
    """

    def __init__(self, planner_config, topo_agent):
        assert isinstance(topo_agent, SloopMosTopo2DAgent)
        self.planner_config = planner_config
        self._topo_agent = topo_agent
        self._mos2d_agent = self._create_mos2d_agent()
        self._subgoal_planner = pomdp_py.POUCT(**self.planner_config['high_level_planner_args'],
                                               rollout_policy=self._topo_agent.policy_model)
        self._subgoal_handler = None
        self._current_subgoal = None

    @property
    def mos2d_agent(self):
        return self._mos2d_agent

    def _create_mos2d_agent(self):
        agent_config = self._topo_agent.agent_config.copy()

        srobot_topo = self._topo_agent.belief.mpe().s(self._topo_agent.robot_id)
        init_robot_state = RobotState(self._topo_agent.robot_id,
                                      srobot_topo.pose,
                                      srobot_topo.objects_found,
                                      srobot_topo.camera_direction)
        init_belief = BeliefBasic2D(self._topo_agent.target_objects,
                                    robot_state=init_robot_state,
                                    object_beliefs=dict(self._topo_agent.belief.object_beliefs))
        agent = MosBasic2DAgent(agent_config,
                                self._topo_agent.grid_map,
                                init_belief=init_belief)

        return agent


    def plan(self, agent):
        """
        Args:
            agent: a topo2d agent.
        """
        # ########## Only plans subgoal when previous one done ########
        # if isinstance(self._current_subgoal, StayAction):
        #     self._current_subgoal = None  # replan if stay

        # if self._current_subgoal is None:
        #     subgoal = self._subgoal_planner.plan(self._topo_agent)
        #     self._current_subgoal = subgoal
        #     print(typ.bold(typ.blue(f"Subgoal planned: {subgoal})")))

        #     self._subgoal_handler = self.handle(self._current_subgoal)

        ########## Always replan subgoal ########
        subgoal = self._subgoal_planner.plan(self._topo_agent)
        print(typ.bold(typ.blue(f"Subgoal planned: {subgoal})")))

        if self._current_subgoal is None:
            self._current_subgoal = subgoal

        if self._subgoal_handler is None:
            self._subgoal_handler = self.handle(subgoal)
        else:
            if subgoal != self._subgoal_handler.subgoal:
                self._subgoal_handler = self.handle(subgoal)

        action = self._subgoal_handler.step()
        return action


    def handle(self, subgoal):
        if isinstance(subgoal, StayAction):
            return LocalSearchHandler(subgoal, self._topo_agent, self._mos2d_agent,
                                      self.planner_config['local_search'])
        elif isinstance(subgoal, MotionActionTopo):
            # set a destination pose; the orientation is determined by the transition
            # model given a random state sampled from belief.
            rnd_state = self._topo_agent.belief.random()
            robot_trans_model = self._topo_agent.transition_model[self._topo_agent.robot_id]
            subgoal.dst_pose = robot_trans_model.sample(rnd_state, subgoal).pose

            # Check whether we want to plan individual movements to fulfill the navigation
            # subgoal, or if we just want to directly output the navigation subgoal
            plan_nav_actions = self.planner_config.get("plan_nav_actions", True)
            if plan_nav_actions:
                return NavTopoHandler(subgoal, self._topo_agent, self._mos2d_agent)
            else:
                return NavTopoIdentityHandler(subgoal, self._topo_agent)

        elif isinstance(subgoal, FindAction):
            return FindHandler(subgoal)

    def update(self, agent, action, observation):
        if self._subgoal_handler is not None:
            self._subgoal_handler.update(action, observation)
            self._subgoal_planner.update(agent, action, observation)

            if self._subgoal_handler.done:
                self._subgoal_handler = None
                self._current_subgoal = None
