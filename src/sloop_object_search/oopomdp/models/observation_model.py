"""Defines the ObservationModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Observation: {objid : pose(x,y) or NULL}. The sensor model could vary;
             it could be a fan-shaped model as the original paper, or
             it could be something else. But the resulting observation
             should be a map from object id to observed pose or NULL (not observed).

Observation Model

  The agent can observe its own state, as well as object poses
  that are within its sensor range. We only need to model object
  observation.

"""

import pomdp_py
import math
import random
import numpy as np
from ..domain.state import *
from ..domain.action import *
from ..domain.observation import *

#### Observation Models ####


### Observation models
class ObjectDetectionModel:
    """
    Models: Pr(zi | si', sr', a);
    Domain-specific.
    """
    def __init__(self, objid):
        self.objid = objid

    def probability(self, zi, si, srobot):
        """
        Args:
            zi: object observation
            si: object state
            srobot: robot state
            action: action taken
        """
        raise NotImplementedError

    def sample(self, si, srobot):
        """
        Args:
            si: object state
            srobot: robot state
            action: action taken
        Returns:
            zi: ObjectDetection
        """
        raise NotImplementedError


class RobotObservationModel:
    """Pr(zrobot | srobot); default is identity"""
    def __init__(self, robot_id):
        self.robot_id = robot_id

    def sample(self, srobot_next, action):
        robotobz = RobotObservation(self.robot_id,
                                    srobot_next['pose'],
                                    srobot_next['objects_found'],
                                    srobot_next['camera_direction'])
        return robotobz

    def probability(self, zrobot, srobot_next, action):
        def robot_state_from_obz(zrobot):
            return RobotState(zrobot.robot_id,
                              zrobot.pose,
                              zrobot.objects_found,
                              zrobot.camera_direction)
        srobot_from_z = robot_state_from_obz(zrobot)
        return identity(srobot_from_z, srobot)


def receiving_observation(action, no_look=False):
    """Returns true if the `action` receives sensing observation"""
    if not no_look:
        return isinstance(action, LookAction)
    else:
        return isinstance(action, MotionAction)


class GMosObservationModel(pomdp_py.OOObservationModel):
    """
    The observation model for multi-object search
    (generalization of the original, as it allows
    other kinds of object detection models and also
    considers correlations)

    Note that spatial language observation is not
    part of this.

    -- Comment on correlations modeling --

    COS_POMDP handled the correlational observation case when
    there is a single target object. In the case where there
    are multiple target objects, the correlational model would
    be (with the conditional independence assumptions made)

       Pr(xi | xtarget1, ... xtargetm)

    which is not very satisfying in the sense that it only
    considers the correlation between an observed object i
    with target objects (which is mathematically correct
    but such a model may be unintuitive to obtain, and
    computationally expensive to maintain in practice).

    In reality, when talking about correlations, for the most
    part, it is about incorporating correlational information
    during belief update.

    Therefore, we only have ObjectDetectionModel.
    """
    def __init__(self, robot_id,
                 detection_models,
                 robot_observation_model=None,
                 correlation_model=None,
                 no_look=False):
        """
        detection_models: maps from objid to ObjectDetectionModel
        robot_obserfvation_model (RobotObservationModel): If None,
           will default to the identity model.
        correlation_model: A joint probability model (probability.JointDist)
            for all detectable objects and objects in state.
        """
        self.robot_id = robot_id
        self.detection_models = detection_models
        self.correlation_model = correlation_model
        self._no_look = no_look
        if self.robot_observation_model is None:
            self.robot_observation_model = RobotObservationModel(robot_id)

        observation_models = {**{robot_id: self.robot_observation_model},
                              **{j: self.detection_models[j]
                                 for j in self.detection_models}}
        super().__init__(self, observation_models)

    def sample(self, snext, action):
        if not receiving_observation(action, no_look=self._no_look):
            return MosOOObservation({})

        srobot_next = snext.s(self.robot_id)
        zrobot = self.robot_observation_model.sample(srobot_next, action)
        objobzs = {self.robot_id: zrobot}
        for j in self.detection_models:
            # j is an object id
            if j in next_state.object_states:
                # we maintain state for object j; no correlation is needed.
                sj_next = snext.s(j)
                zj = self.detection_models[j]\
                     .sample(sj_next, srobot_next, action)
            else:
                # sdist_j: state distribution for object j; Even though the
                # state of j is not maintained in next_state, it can be sampled
                # based on correlations.
                try:
                    sdist_j =\
                        self.correlation_model.marginal(
                            [j], evidence=snext.object_states)
                    sj_next = sdist_j.sample()
                    zj = self.detection_models[j].sample(sj_next, srobot_next)
                except (KeyError, ValueError, AttributeError):
                    zj = ObjectDetection(j, ObjectDetection.NULL)
            objobzs[j] = zj
        return MosOOObservation(objobzs)

    def probability(self, observation, snext, action):
        """
        observation (JointObservation)
        """
        if not receiving_observation(action, no_look=self._no_look):
            # No observation should be received
            if observation.pose == ObjectObservation.NULL:
                return 1.0
            else:
                return 0.0

        zrobot = observation.z(self.robot_id)
        srobot_next = snext.s(self.robot_id)
        pr_zrobot = self.robot_observation_model.probability(
            zrobot, srobot_next, action)
        pr_joint = 1.0 * pr_zrobot
        for j in self.detection_models:
            if j not in observation:
                zj = ObjectDetection(j, ObjectDetection.NULL)
            else:
                zj = observation.z(j)
            if j in snext.object_states:
                sj_next = snext.s(j)
                pr_zj = self.detection_models[j].probability(zj, sj_next, srobot_next)
            else:
                sdist_j =\
                    self.correlation_model.marginal(
                        [j], evidence=snext.object_states)
                pr_zj = 1e-12
                for sj_next in sdist_j:
                    pr_detection = self.detection_models[j].probability(zj, sj_next, srobot_next)
                    pr_corr = sdist_j.prob({j:sj_next})
                    pr_zj += pr_detection*pr_corr
            pr_joint *= pr_zj
        return pr_joint









# #################################################################################################
# def receiving_observation(action, no_look=False):
#     """Returns true if the `action` receives sensing observation"""
#     if not no_look:
#         return isinstance(action, LookAction)
#     else:
#         return isinstance(action, MotionAction)


# class MosObservationModel(pomdp_py.OOObservationModel):
#     """Object-oriented transition model"""
#     def __init__(self,
#                  dim,
#                  sensor,
#                  object_ids,
#                  sigma=0.01,
#                  epsilon=1,
#                  no_look=False):
#         self.sigma = sigma
#         self.epsilon = epsilon
#         self._no_look = no_look
#         observation_models = {objid: ObjectObservationModel(objid, sensor, dim,
#                                                             sigma=sigma, epsilon=epsilon,
#                                                             no_look=no_look)
#                               for objid in object_ids}
#         pomdp_py.OOObservationModel.__init__(self, observation_models)

#     def sample(self, next_state, action, argmax=False, **kwargs):
#         if not receiving_observation(action, no_look=self._no_look):
#             return MosOOObservation({})

#         factored_observations = super().sample(next_state, action, argmax=argmax)
#         return MosOOObservation.merge(factored_observations, next_state)

# class ObjectObservationModel(pomdp_py.ObservationModel):
#     def __init__(self, objid, sensor, dim, sigma=0, epsilon=1, no_look=False):
#         """
#         sigma and epsilon are parameters of the observation model (see paper),
#         dim (tuple): a tuple (width, length) for the dimension of the world"""
#         self._objid = objid
#         self._sensor = sensor
#         self.sigma = sigma
#         self.epsilon = epsilon
#         self._no_look = no_look

#     def _compute_params(self, object_in_sensing_region):
#         if object_in_sensing_region:
#             # Object is in the sensing region
#             alpha = self.epsilon
#             beta = (1.0 - self.epsilon) / 2.0
#             gamma = (1.0 - self.epsilon) / 2.0
#         else:
#             # Object is not in the sensing region.
#             alpha = (1.0 - self.epsilon) / 2.0
#             beta = (1.0 - self.epsilon) / 2.0
#             gamma = self.epsilon
#         return alpha, beta, gamma

#     def probability(self, observation, next_state, action, **kwargs):
#         """
#         Returns the probability of Pr (observation | next_state, action).

#         Args:
#             observation (ObjectObservation)
#             next_state (State)
#             action (Action)
#         """
#         if not receiving_observation(action, no_look=self._no_look):
#             # No observation should be received
#             if observation.pose == ObjectObservation.NULL:
#                 return 1.0
#             else:
#                 return 0.0

#         if observation.objid != self._objid:
#             raise ValueError("The observation is not about the same object")

#         # The (funny) business of allowing histogram belief update using O(oi|si',sr',a).
#         next_robot_state = kwargs.get("next_robot_state", None)
#         if next_robot_state is not None:
#             assert next_robot_state["id"] == self._sensor.robot_id,\
#                 "Robot id of observation model mismatch with given state"
#             robot_pose = next_robot_state.pose

#             if isinstance(next_state, ObjectState):
#                 assert next_state["id"] == self._objid,\
#                     "Object id of observation model mismatch with given state"
#                 object_pose = next_state.pose
#             else:
#                 object_pose = next_state.pose(self._objid)
#         else:
#             robot_pose = next_state.pose(self._sensor.robot_id)
#             object_pose = next_state.pose(self._objid)

#         # Compute the probability
#         zi = observation.pose
#         alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))
#         # Requires Python >= 3.6
#         event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
#         if event_occured == "A":
#             # object in sensing region and observation comes from object i
#             if zi == ObjectObservation.NULL:
#                 # Even though event A occurred, the observation is NULL.
#                 # This has 0.0 probability.
#                 return 1e-9 * alpha
#             else:
#                 gaussian = pomdp_py.Gaussian(list(object_pose),
#                                              [[self.sigma**2, 0],
#                                               [0, self.sigma**2]])
#                 return gaussian[zi] * alpha
#         elif event_occured == "B":
#             return (1.0 / self._sensor.sensing_region_size) * beta

#         else: # event_occured == "C":
#             prob = 1.0 if zi == ObjectObservation.NULL else 1e-9  # indicator zi == NULL
#             return prob * gamma


#     def sample(self, next_state, action, **kwargs):
#         """Returns observation"""
#         if not receiving_observation(action, no_look=self._no_look):
#             # No observation should be received
#             return ObjectObservation(self._objid, ObjectObservation.NULL)

#         robot_pose = next_state.pose(self._sensor.robot_id)
#         object_pose = next_state.pose(self._objid)

#         # Obtain observation according to distribution.
#         alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))

#         # Requires Python >= 3.6
#         event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
#         zi = self._sample_zi(event_occured, next_state)

#         return ObjectObservation(self._objid, zi)

#     def argmax(self, next_state, action, **kwargs):
#         # Obtain observation according to distribution.
#         alpha, beta, gamma = self._compute_params(self._sensor.within_range(robot_pose, object_pose))

#         event_probs = {"A": alpha,
#                        "B": beta,
#                        "C": gamma}
#         event_occured = max(event_probs, key=lambda e: event_probs[e])
#         zi = self._sample_zi(event_occured, next_state, argmax=True)
#         return ObjectObservation(self._objid, zi)

#     def _sample_zi(self, event, next_state, argmax=False):
#         if event == "A":
#             object_true_pose = next_state.object_pose(self._objid)
#             gaussian =  pomdp_py.Gaussian(list(object_true_pose),
#                                           [[self.sigma**2, 0],
#                                            [0, self.sigma**2]])
#             if not argmax:
#                 zi = gaussian.random()
#             else:
#                 zi = gaussian.mpe()
#             zi = (int(round(zi[0])), int(round(zi[1])))

#         elif event == "B":
#             zi = (random.randint(0, self._gridworld.width),   # x axis
#                   random.randint(0, self._gridworld.height))  # y axis
#         else: # event == C
#             zi = ObjectObservation.NULL
#         return zi
