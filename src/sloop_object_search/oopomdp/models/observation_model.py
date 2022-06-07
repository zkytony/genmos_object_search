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
from ..domain.state import (RobotState,
                            ObjectState)
from ..domain.action import (MotionAction,
                             LookAction,
                             FindAction)
from ..domain.observation import (ObjectDetection,
                                  ObjectDetection2D,
                                  RobotObservation,
                                  RobotObservation2D,
                                  RobotObservationTopo,
                                  GMOSObservation)


### Observation models
class ObjectDetectionModel:
    """
    Models: Pr(zi | si', sr', a);
    Domain-specific.
    """
    def __init__(self, objid, *args):
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

    @property
    def observation_class(self):
        raise NotImplementedError


class RobotObservationModel(pomdp_py.ObservationModel):
    """Pr(zrobot | srobot); default is identity"""
    def __init__(self, robot_id):
        self.robot_id = robot_id

    @property
    def observation_class(self):
        return RobotObservation

    def sample(self, snext, action):
        srobot_next = snext.s(self.robot_id)
        robotobz = self.observation_class.from_state(srobot_next)
        return robotobz

    def probability(self, zrobot, snext, action):
        srobot_next = snext.s(self.robot_id)
        srobot_from_z = srobot_next.__class__.from_obz(zrobot)
        return identity(srobot_from_z, srobot)


class RobotObservationModel2D(RobotObservationModel):
    @property
    def observation_class(self):
        return RobotObservation2D

class RobotObservationModelTopo(RobotObservationModel):
    @property
    def observation_class(self):
        return RobotObservationTopo


def receiving_observation(action, no_look=False):
    """Returns true if the `action` receives sensing observation"""
    if not no_look:
        return isinstance(action, LookAction)
    else:
        return True


class ObjectObservationModel(pomdp_py.ObservationModel):
    """
    ObjectObservationModel is an ObservationModel;
    Note that ObjectDetectionModel is not.
    Note that correlation is not considered here; you
    may do that during belief update.
    """
    def __init__(self, objid, robot_id, detection_model, no_look=False):
        self.objid = objid
        self.robot_id = robot_id
        self.detection_model = detection_model
        self._no_look = no_look

    def sample(self, snext, action):
        """
        Just sample observation for THIS object
        """
        if not receiving_observation(action, no_look=self._no_look):
            return self.detection_model.observation_class.null_observation(self.objid)
        srobot_next = snext.s(self.robot_id)
        sobj_next = snext.s(self.objid)
        zobj = self.detection_model.sample(sobj_next, srobot_next)
        return zobj

    def probability(self, zobj, snext, action):
        """
        observation (ObjectDetection)
        """
        if not receiving_observation(action, no_look=self._no_look):
            # No observation should be received
            if zobj.pose == self.detection_model.observation_class.NULL:
                return 1.0
            else:
                return 0.0

        srobot_next = snext.s(self.robot_id)
        sobj_next = snext.s(self.objid)
        return self.detection_model.probability(zobj, sobj_next, srobot_next)


class GMOSObservationModel(pomdp_py.OOObservationModel):
    """
    GMOS = generalized MOS

    The observation model for multi-object search
    (generalization of the original, as it allows
    other kinds of object detection models and also
    considers correlations)

    Note that correlation is not considered here; you
    may do that during belief update.
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
        self.robot_observation_model = robot_observation_model
        if self.robot_observation_model is None:
            self.robot_observation_model = RobotObservationModel(robot_id)

        observation_models = {**{robot_id: self.robot_observation_model},
                              **{j: ObjectObservationModel(
                                  j, self.robot_id, self.detection_models[j], no_look=no_look)
                                 for j in self.detection_models}}
        super().__init__(observation_models)

    def sample(self, snext, action):
        factored_observations = super().sample(snext, action)
        return GMOSObservation(factored_observations)
