# Defines the agent. There's nothing special
# about the MOS agent in fact, except that
# it uses models defined in ..models, and
# makes use of the belief initialization
# functions in belief.py
import pomdp_py
from .belief import *
from ..models.transition_model import *
from ..models.observation_model import *
from ..models.reward_model import *
from ..models.policy_model import *
from ..domain.action import MOTION_SCHEME

class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""
    def __init__(self,
                 robot_id,
                 target_object_ids,
                 init_robot_state,  # initial robot state (assuming robot state is observable perfectly)
                 search_region,
                 robot_trans_model,
                 policy_model,
                 detectors,
                 reward_model,
                 targets_belief_initializer,
                 targets_belief_updater,
                 belief_type="histogram",
                 prior={},
                 binit_args={},
                 use_heuristic=True,
                 no_look=False):
        """
        Args:
            robot_id (any hashable)
            target_object_ids (array-like): list of target object ids
            init_robot_state (RobotState)
            search_region (SearchRegion): possible locations for the target
            robot_trans_model (RobotTransModel): transition model for the robot
            policy_model (PolicyModel): policy model
            detectors: Maps from DetectionModel Pr(zi | si, srobot')
                Must contain an entry for the target object
            belief_type: type of belief representation.
            prior: Maps from search region location to a float.
        # TODO: Add correlations
        """
        init_brobot = self._initialize_robot_belief(init_robot_state)
        init_bobjects = targets_belief_initializer(target_object_ids, search_region,
                                                   belief_type, prior,
                                                   init_robot_state, **binit_args)
        init_belief = MosJointBelief({**{robot_id: init_brobot},
                                      **init_bobjects})
        self.targets_belief_updater = targets_belief_initializer
        transition_model = MosTransitionModel(target_object_ids,
                                              robot_trans_model)
        observation_model = MosObservationModel(...)
        policy_model.set_observation_model(observation_model,
                                           use_heuristic=use_heuristic)
        super().__init__(init_belief, policy_model,
                         transition_model, observation_model, reward_model)


    def _initialize_robot_belief(self, init_robot_state):
        """The robot state is known"""
        return pomdp_py.Histogram({init_robot_state: 1.0})













                 object_ids,  # target object ids
                 dim,         # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
                 detection_models,      # Sensor equipped on the robot
                 belief_rep="histogram",  # belief representation, either "histogram" or "particles".
                 prior={},       # prior belief, as defined in belief.py:initialize_belief
                 num_particles=100,  # used if the belief representation is particles
                 grid_map=None,    # GridMap used to avoid collision with obstacles (None if not provided)
                 no_look=False,
                 reward_small=1):   # Agent doesn't have look action -- an observation is received per move.
        self.robot_id = robot_id
        self._object_ids = object_ids

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]

        # initialize belief
        init_belief = initialize_belief(dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={self.robot_id:rth},
                                        num_particles=num_particles)
        transition_model = GMosTransitionModel(dim,
                                               detection_models,
                                               self._object_ids,
                                               no_look=no_look)
        observation_model = GMosObservationModel(self.robot_id,
                                                 detection_models,
                                                 no_look=no_look)
        reward_model = GoalRewardModel(self._object_ids,
                                       robot_id=self.robot_id,
                                       small=reward_small)
        if MOTION_SCHEME == "xy":
            action_prior = GreedyActionPriorXY(self.robot_id,
                                               grid_map,
                                               10, reward_model.big,
                                               no_look=no_look)
        elif MOTION_SCHEME == "vw":
            action_prior = GreedyActionPriorVW(self.robot_id,
                                               grid_map,
                                               10, reward_model.big,
                                               no_look=no_look)
        policy_model = PreferredPolicyModel(action_prior)
        # policy_model = PolicyModel(self.robot_id, no_look=no_look)
        super().__init__(init_belief, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None

    @property
    def grid_map(self):
        return self.policy_model.grid_map
