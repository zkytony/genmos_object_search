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
