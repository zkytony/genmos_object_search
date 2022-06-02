import pomdp_py
from ..domain.state import RobotState
from ..domain.observation import (JointObservation,
                                  RobotObservation,
                                  ObjectDetection)
from ..utils import identity

### Observation models
class ObjectDetectionModel:
    """
    Models: Pr(zi | si', sr', a);
    Domain-specific.
    """
    def __init__(self, objid):
        self.objid = objid

    def probability(self, zi, si, srobot, action):
        """
        Args:
            zi: object observation
            si: object state
            srobot: robot state
            action: action taken
        """
        raise NotImplementedError

    def sample(self, si, srobot, action):
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


class MosObservationModel(pomdp_py.OOObservationModel):
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

    Therefore, in this SLOOP generic model, we only have
    ObjectDetectionModel.

    If you would like to consider object correlations during
    planning, there is a way to do so with this framework.
    Basically, it is possible for an object observation to
    be correlated with other object observations. As a result,
    those observations z* will be accessible in the sample function.
    """
    def __init__(self, robot_id,
                 detection_models,
                 robot_observation_model=None,
                 correlation_model=None):
        """
        detection_models: maps from objid to ObjectDetectionModel
        robot_obserfvation_model (RobotObservationModel): If None,
           will default to the identity model.
        correlation_model: A joint probability model (JointDist) for all detectable
            objects and objects in state.
        """
        self.robot_id = robot_id
        self.detection_models = detection_models
        self.correlation_model = correlation_model
        if self.robot_observation_model is None:
            self.robot_observation_model = RobotObservationModel(robot_id)

        observation_models = {**{robot_id: self.robot_observation_model},
                              **{j: self.detection_models[j]
                                 for j in self.detection_models}}
        super().__init__(self, observation_models)

    def sample(self, snext, action):
        srobot_next = snext.s(self.robot_id)
        zrobot = self.robot_observation_model.sample(srobot_next, action)
        objobzs = {self.robot_id: zrobot}
        for j in self.detection_models:
            # j is an object id
            if j in next_state.object_states:
                sj_next = snext.s(j)
                zj = self.detection_models[j]\
                     .sample(sj_next, srobot_next, action)
            else:
                # sdist_j: state distribution for object j; Even though the
                # state of j is not maintained in next_state, it can be sampled
                # based on correlations. Note: not tested yet.
                try:
                    sdist_j =\
                        self.correlation_model.marginal(
                            [j], evidence=snext.object_states)
                    sj_next = sdist_j.sample()
                    zj = self.detection_models[j]\
                             .sample(sj_next, srobot_next, action)
                except (KeyError, ValueError, AttributeError):
                    zj = ObjectDetection(j, ObjectDetection.NULL)
            objobzs[j] = zj
        return JointObservation(objobzs)

    def probability(self, observation, snext, action):
        """
        observation (JointObservation)
        """
        zrobot = observation.z(self.robot_id)
        srobot_next = snext.s(self.robot_id)
        pr_zrobot = self.robot_observation_model.probability(
            zrobot, srobot_next, action)
        for j in self.detection_models:
            if j not in observation:
                zj = ObjectDetection(j, ObjectDetection.NULL)
