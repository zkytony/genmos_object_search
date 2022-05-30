import rospy
import actionlib
import pomdp_py
from sloop_ros.utils.misc import tobeoverriden
from sloop_ros.msg import (PlanNextStepAction,
                           PlanNextStepResult,
                           DefaultAction,
                           DefaultObservation,
                           DefaultBelief)
from actionlib_msgs.msg import GoalStatus


class BaseAgent(pomdp_py.Agent):
    """
    A base agent serves:
    - /plan

    subscribes to:
    - /observation

    publishes:
    - /action
    - /belief
    """
    def __init__(self, belief, models, **kwargs):
        super().__init__(belief,
                         models.policy_model,
                         transition_model=models.transition_model,
                         observation_model=models.observation_model,
                         reward_model=models.reward_model)
        self._planner = None

        # Note that the following objects are created when 'setup' is called.
        # If you would like the agent
        self._plan_server = None   # calls the service to plan the next step
        self._action_publisher = None
        self._belief_publisher = None
        self._observation_subscriber = None

        self._plan_service = kwargs.get("plan_service", "plan")
        self._action_topic = kwargs.get("action_topic", "action")
        self._belief_topic = kwargs.get("belief_topic", "belief")
        self._observation_topic = kwargs.get("observation_topic", "observation")

        self._belief_rate = kwargs.get("belief_publish_rate", 5)  # Hz

        # This will always be equal to the last planned action
        self._last_action = None


    @tobeoverriden
    def setup(self):
        """Override this function to create make your agent
        work with different message types."""
        # Action server for DoPlan (perform one planning step)
        self._plan_server = actionlib.SimpleActionServer(
            self._plan_service, PlanNextStepAction, self.plan)

        # Publishes most recent action
        self._action_publisher = rospy.Publisher(
            self._action_topic, DefaultAction, queue_size=10, latch=True)

        # Publishes current belief
        self._belief_publisher = rospy.Publisher(
            self._belief_topic, DefaultBelief, queue_size=10, latch=True)

        # Subscribes to observation
        self._observation_subscriber = rospy.Subscriber(
            self._observation_topic, DefaultObservation, self._observation_cb)


    @tobeoverriden
    def _observation_cb(self, observation_msg):
        """Override this function to handle different observation types"""
        rospy.loginfo(f"Observation received: {observation}")
        observation = self.interpret_observation_msg(observation_msg)
        self.belief.update(self, observation, self._last_action)


    @tobeoverriden
    def interpret_observation_msg(self, observation_msg):
        """Given a ROS message, return a pomdp_py.Observation"""
        return pomdp_py.SimpleObservation(observation_msg.data)


    def run(self):
        """Blocking call"""
        if self._plan_server is None\
           or self._action_publisher is None\
           or self._belief_publisher is None\
           or self._observation_subscriber is None:
            rospy.logerr("Unable to run. {}\n Did you run 'setup'?"
                         .format(self._setup_help_message()))
            return

        self._plan_server.start()
        belief_msg = self.belief.to_ros_msg()
        rospy.Timer(rospy.Duration(1./self._belief_rate),
                    lambda event: self._belief_publisher.publish(belief_msg))
        rospy.loginfo("Running agent {}".format(self.__class__.__name__))
        rospy.spin()


    def plan(self, goal):
        result = PlanNextStepResult()
        if self._planner is None:
            rospy.logerr("Agent's planner is not set. Cannot plan.")
            result.status = GoalStatus.REJECTED
            self.plan_server.set_rejected(result)
            self._last_action = None
        else:
            action = self._planner.plan(self.belief)
            action_msg = action.to_ros_msg(goal.goal_id)
            rospy.loginfo(f"Planning successful. Action: {action}")
            rospy.loginfo("Action published")
            self._last_action = action
            self._action_publisher.publish(action_msg)
            result.status = GoalStatus.SUCCEEDED
            self.plan_server.set_succeeded(result)


    def set_planner(self, planner):
        self._planner = planner


    def _setup_help_message(self):
        message = "The following objects should not be None:\n"
        if self._plan_server is None:
            message += "- self._plan_server\n"
        if self._action_publisher is None:
            message += "- self._action_publisher\n"
        if self._belief_publisher is None:
            message += "- self._belief_publisher\n"
        if self._observation_subscriber is None:
            message += "- self._observation_subscriber\n"
        return message
