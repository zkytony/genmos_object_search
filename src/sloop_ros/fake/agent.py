import rospy
import pomdp_py
from sloop_ros.core import BaseAgent

class FakeAgent(BaseAgent):
    def setup(self):
        print("Setting up Fake Agent!")
