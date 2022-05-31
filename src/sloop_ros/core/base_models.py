import pomdp_py
from sloop_ros.msg import (DefaultObservation,
                           DefaultAction)
from sloop_ros.utils.misc import tobeoverriden

class BaseObservationModel(pomdp_py.ObservationModel):
    @property
    @tobeoverriden
    def ros_observation_msg_type(self):
        return DefaultObservation

    @tobeoverriden
    def interpret_observation_msg(self, observation_msg):
        return pomdp_py.SimpleObservation(observation_msg.data)


class BasePolicyModel(pomdp_py.RolloutPolicy):
    @property
    def ros_action_msg_type(self):
        return DefaultAction
