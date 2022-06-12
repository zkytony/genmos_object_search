# Hierarchical agent, with a hierarchical planner
from sloop.agent import SloopAgent
from ..agent.topo2d import SloopMosTopo2DAgent
from ..agent.basic2d import MosBasic2DAgent
from ..models.belief import BeliefBasic2D
from ..domain.state import RobotState2D
from ..domain.action import MotionActionTopo, StayAction, FindAction
from .handlers import LocalSearchHandler, NavTopoHandler, FindHandler

import pomdp_py

class HierarchicalPlanner(pomdp_py.Planner):

    def __init__(self, planner_config, topo_agent):
        assert isinstance(topo_agent, SloopMosTopo2DAgent)
        self.planner_config = planner_config
        self._topo_agent = topo_agent
        self._mos2d_agent = self._create_mos2d_agent()
        self._subgoal_planner = pomdp_py.POUCT(**self.planner_config['subgoal_level'],
                                               rollout_policy=self._topo_agent.policy_model)
        self._subgoal_handler = None

    @property
    def mos2d_agent(self):
        return self._mos2d_agent

    def _create_mos2d_agent(self):
        agent_config = self._topo_agent.agent_config.copy()

        srobot_topo = self._topo_agent.belief.mpe().s(self._topo_agent.robot_id)
        init_robot_state = RobotState2D(self._topo_agent.robot_id,
                                        srobot_topo.pose,
                                        srobot_topo.objects_found,
                                        srobot_topo.camera_direction)
        init_belief = BeliefBasic2D(init_robot_state,
                                    self._topo_agent.target_objects,
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
        if self._subgoal_handler is None:
            subgoal = self._subgoal_planner.plan(self._topo_agent)
            self._subgoal_handler = self.handle(subgoal)

        action = self._subgoal_handler.step()
        return action


    def handle(self, subgoal):
        if isinstance(subgoal, StayAction):
            return LocalSearchHandler(subgoal, self._topo_agent, self._mos2d_agent,
                                      self.planner_config['local_search'])
        elif isinstance(subgoal, MotionActionTopo):
            rnd_state = self._topo_agent.belief.random()
            robot_trans_model = self._topo_agent.transition_model[self._topo_agent.robot_id]
            subgoal.dst_pose = robot_trans_model.sample(rnd_state, subgoal).pose
            return NavTopoHandler(subgoal, self._topo_agent, self._mos2d_agent)

        elif isinstance(subgoal, FindAction):
            return FindHandler(subgoal)

    def update(self, agent, action, observation):
        if self._subgoal_handler is not None:
            self._subgoal_handler.update(action, observation)
            self._subgoal_planner.update(agent, action, observation)
