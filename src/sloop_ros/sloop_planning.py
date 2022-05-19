# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The sloop_planner is responsible for:
#
# (1) creating the POMDP Agent for the SLOOP task
# (2) listening to belief and then use that to produce the next action
# (3) publish the next action
import rospy
from std_msgs.msg import String

def belief_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    sloop_plan_next()

def sloop_plan_next():
    print("Planning next action")

def sloop_planner():
    rospy.init_node('sloop_planner', anonymous=True)
    agent = "agent"

    sloop_belief_topic = rospy.get_param("~belief_topic")
    belief_sub = rospy.Subscriber(sloop_belief_topic, String, belief_callback)

    rospy.spin()
