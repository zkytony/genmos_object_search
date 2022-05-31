import rospy
import pomdp_py
from sloop_ros.msg import DefaultBelief
from sloop_ros.utils.misc import tobeoverriden


class BaseBelief(pomdp_py.GenerativeDistribution):

    @tobeoverriden
    def update(self, agent, observation, action):
        pass

    @tobeoverriden
    def to_ros_msg(self, stamp=None):
        belief_msg = DefaultBelief()
        if stamp is None:
            stamp = rospy.Time.now()
        belief_msg.stamp = stamp
        belief_msg.states = [state for state in self]
        belief_msg.probs = [self[state]
                            for state in belief_msg.states]
        return belief_msg

    @property
    @tobeoverriden
    def ros_belief_msg_type(self):
        return DefaultBelief
