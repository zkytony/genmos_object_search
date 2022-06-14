import rospy
import actionlib
import pomdp_py
from actionlib_msgs.msg import GoalStatus
from sloop_ros.msg import (PlanNextStepAction,
                           PlanNextStepResult,
                           DefaultAction,
                           DefaultBelief,
                           DefaultObservation)
from sloop_object_search.utils.misc import import_class


class BaseAgentWrapper:
    """
    A base agent serves:
    - ~plan

    subscribes to:
    - ~observation

    publishes:
    - ~action
    - ~belief

    See scripts/run_pomdp_agent for how this is used.
    """
    def __init__(self, pomdp_agent, ros_config={}, planner=None):
        self.agent = pomdp_agent
        self._planner = planner
        self._ros_config = ros_config

        # Note that the following objects are created when 'setup' is called.
        # If you would like the agent
        self._plan_server = None   # calls the service to plan the next step
        self._action_publisher = None
        self._belief_publisher = None
        self._observation_subscriber = None

        self._plan_service = self._ros_config.get("plan_service", "~plan")  # would become <node_name>/plan
        self._action_topic = self._ros_config.get("action_topic", "~action")
        self._belief_topic = self._ros_config.get("belief_topic", "~belief")
        self._observation_topic = self._ros_config.get("observation_topic", "~observation")

        self._belief_rate = self._ros_config.get("belief_publish_rate", 5)  # Hz

        # This will always be equal to the last planned action
        self._last_action = None


    def setup(self):
        """Override this function to create make your agent
        work with different message types."""
        # Action server for DoPlan (perform one planning step)
        self._plan_server = actionlib.SimpleActionServer(
            self._plan_service, PlanNextStepAction, self.plan, auto_start=False)

        # Publishes most recent action
        action_msg_type = DefaultAction
        if "action_msg_type" in self._ros_config:
            action_msg_type = import_class(self._ros_config["action_msg_type"])
        self._action_publisher = rospy.Publisher(
            self._action_topic,
            action_msg_type,
            queue_size=10, latch=True)

        # Publishes current belief
        belief_msg_type = DefaultBelief
        if "belief_msg_type" in self._ros_config:
            belief_msg_type = self._ros_config["belief_msg_type"]
        self._belief_publisher = rospy.Publisher(
            self._belief_topic,
            belief_msg_type,
            queue_size=10, latch=True)

        # Subscribes to observation
        observation_msg_type = DefaultObservation
        if "observation_msg_type" in self._ros_config:
            observation_msg_type = self._ros_config["observation_msg_type"]
        self._observation_subscriber = rospy.Subscriber(
            self._observation_topic,
            observation_msg_type,
            self._observation_cb)

    def _observation_cb(self, observation_msg):
        """Override this function to handle different observation types"""
        rospy.loginfo(f"Observation received: {observation_msg}")
        observation = self.observation_model.interpret_observation_msg(observation_msg)
        self.agent.belief.update(self, observation, self._last_action)

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
        belief_msg = self.belief_to_ros_msg(self.agent.belief)
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
            action = self._planner.plan(self)
            rospy.loginfo(f"Planning successful. Action: {action}")
            rospy.loginfo("Action published")
            action_msg = self.action_to_ros_msg(action, goal.goal_id)
            self._action_publisher.publish(action_msg)
            self._last_action = action
            result.status = GoalStatus.SUCCEEDED
            self._plan_server.set_succeeded(result)

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

    def belief_to_ros_msg(self, belief, stamp=None):
        """To Be Overriden"""
        belief_msg = DefaultBelief()
        # if stamp is None:
        #     stamp = rospy.Time.now()
        # belief_msg.stamp = stamp
        # belief_msg.states = [state for state in belief]
        # belief_msg.probs = [belief[state]
        #                     for state in belief_msg.states]
        return belief_msg

    def action_to_ros_msg(self, action, goal_id):
        """To Be Overriden"""
        # All Action message types are assumed to have a goal_id
        # (actionlib goal)
        action_msg = DefaultAction()
        # action_msg.goal_id = goal_id
        # action_msg.name = action.name
        # action_msg.data = str(action.data)
        return action_msg

    def interpret_observation_msg(self, observation, goal_id):
        """To Be Overriden"""
        # All Action message types are assumed to have a goal_id
        # (actionlib goal)
        return pomdp_py.SimpleObservation(observation_msg.data)
