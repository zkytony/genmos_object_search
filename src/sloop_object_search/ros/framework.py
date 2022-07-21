import rospy
import tf2_ros
import actionlib
import pomdp_py
import std_msgs.msg as std_msgs
from pomdp_py.utils import typ
from actionlib_msgs.msg import GoalStatus
from sloop_ros.msg import (PlanNextStepAction,
                           PlanNextStepResult,
                           KeyValAction,
                           DefaultBelief,
                           DefaultObservation)
from sloop_object_search.utils.misc import import_class
from . import ros_utils


class BaseAgentROSBridge:
    """
    Builds a bridge between POMDP agent and ROS.
    (note: this has nothing to do with rosbridge)

    A base agent serves:
    - ~plan

    subscribes to:
    - ~observation/{observation_type}

    publishes:
    - ~action
    - ~belief  (for visualization)
    - ~progress    (for checking task progress; String type)

    Notes:
    1. Observation types are specified through configuration:

       observation:
           <observation_type>
               msg_type: import path of the ROS message type for this observation

    2. Note that this is robot-independent; Several aspects needs to be robot-specific.
       Specifically, in the ros_config, there should be:

       - action_executor: import path of the action executor class
       - observation interpreter: import path of the observation interpreter class

       The action executor will be ran separately as a node. The implementation of this
       class only uses the action executor's classmethod to convert a POMDP action to
       a ROS message.

       The observation interpreter provides the callback function for an observation type.
       Its job is to interpret an observation and update the agent's belief.

    3. The planned action will be published to the ~action topic, but the actual execution
       will be carried out by calling a service provided by the action executor class.
    """
    def __init__(self, ros_config={}, planner=None):
        self._agent = None
        self._planner = planner
        self._ros_config = ros_config

        # Note that the following objects are created when 'setup' is called.
        # If you would like the agent
        self._plan_server = None   # calls the service to plan the next step
        self._action_publisher = None
        self._observation_subscribers = {}

        self._plan_service = self._ros_config.get("plan_service", "~plan")  # would become <node_name>/plan
        self._action_topic = self._ros_config.get("action_topic", "~action")
        self._belief_topic = self._ros_config.get("belief_topic", "~belief")

        self._progress_topic = self._ros_config.get("progress_topic", "~progress")
        self.map_frame = self._ros_config.get("map_frame", "graphnav_map")

        self._last_action = None
        self._last_action_msg = None
        self._last_action_status = None
        self._is_planning = False

        self._belief_msg_type = DefaultBelief
        if "belief_msg_type" in self._ros_config:
            self._belief_msg_type = import_class(self._ros_config["belief_msg_type"])
        self._belief_rate = self._ros_config.get("belief_publish_rate", 5)  # Hz

        self._observation_topics = {}
        self._observation_msg_types = {}
        for z_type in ros_config.get("observation", []):
            z_topic = ros_config["observation"][z_type].get("topic", z_type)
            self._observation_topics[z_type] = f"~observation/{z_type}"
            z_msg_type = DefaultObservation
            if "msg_type" in self._ros_config['observation'][z_type]:
                z_msg_type = import_class(self._ros_config['observation'][z_type]["msg_type"])
            self._observation_msg_types[z_type] = z_msg_type

        # Action executor class: it informs how to convert actions to ROS messages.
        self._action_executor_class = import_class(self._ros_config["action_executor"])

        # topic to subscribe for action execution status, published by ActionExecutor
        self._action_exec_status_topic = self._ros_config.get("action_status_topic", "~action_status")

        # Observation interpretor: it informs how to convert observations to ROS messages
        self._observation_interpretor_class = import_class(self._ros_config["observation_interpreter"])

        # tf; need to create listener early enough before looking up to let tf propagate into buffer
        # reference: https://answers.ros.org/question/292096/right_arm_base_link-passed-to-lookuptransform-argument-target_frame-does-not-exist/
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)

    @property
    def agent(self):
        return self._agent

    def set_agent(self, agent):
        self._agent = agent

    def setup(self):
        """Override this function to create make your agent
        work with different message types."""
        # Action server for DoPlan (perform one planning step)
        self._plan_server = actionlib.SimpleActionServer(
            self._plan_service, PlanNextStepAction, self.plan, auto_start=False)

        # Publishes most recent action
        action_msg_type = KeyValAction
        if "action_msg_type" in self._ros_config:
            action_msg_type = import_class(self._ros_config["action_msg_type"])
        self._action_publisher = rospy.Publisher(
            self._action_topic,
            action_msg_type,
            queue_size=10, latch=True)

        # Publishes current belief
        self._belief_publisher = rospy.Publisher(
            self._belief_topic, self._belief_msg_type, queue_size=10, latch=True)

        # Publishes task progress
        self._progress_publisher = rospy.Publisher(
            self._progress_topic, std_msgs.String, queue_size=10)

        # Subscribes to observation types
        for z_type in self._observation_topics:
            self._observation_subscribers[z_type] = rospy.Subscriber(
                self._observation_topics[z_type],
                self._observation_msg_types[z_type],
                self._observation_interpretor_class.get_observation_callback(z_type),
                callback_args=self)

        # subscribes to action executor status
        self._action_exec_status_sub = rospy.Subscriber(
            self._action_exec_status_topic, GoalStatus,
            self._action_exec_status_cb)

        rospy.loginfo(self._setup_info_message())

    def run(self):
        """Blocking call"""
        if self.agent is None:
            raise ValueError("agent not yet created")

        if self._plan_server is None\
           or self._action_publisher is None\
           or self._belief_publisher is None\
           or len(self._observation_subscribers) == 0:
            rospy.logerr("Unable to run. {}\n Did you run 'setup'?"
                         .format(self._setup_troubleshooting_message()))
            return

        self._plan_server.start()

        rospy.loginfo("Running agent {}".format(self.__class__.__name__))
        rate = rospy.Rate(self._belief_rate)
        while not rospy.is_shutdown():
            belief_msg = self.belief_to_ros_msg(self.agent.belief)
            self._belief_publisher.publish(belief_msg)
            progress_msg = self.current_progress_msg()
            self._progress_publisher.publish(progress_msg)
            rate.sleep()

    @property
    def planner(self):
        return self._planner

    def set_planner(self, planner):
        self._planner = planner

    def plan(self, goal):
        result = PlanNextStepResult()
        if self._planner is None:
            rospy.logerr("Agent's planner is not set. Cannot plan.")
            result.status = GoalStatus(status=GoalStatus.REJECTED,
                                       text="Agent's planner is not set. Cannot plan.")
            self._plan_server.set_aborted(result)
            return

        if self._is_planning:
            rospy.logerr("Planner is currently planning. Please wait.")
            result.status = GoalStatus(status=GoalStatus.REJECTED,
                                       text="Planner is currently planning. Please wait.")
            self._plan_server.set_aborted(result)
            return

        if self._last_action is not None:
            rospy.logwarn(f"last action {self._last_action} is still executing")
            result.status = GoalStatus(status=GoalStatus.ABORTED,
                                       text=f"last action {self._last_action} is still executing")
            self._plan_server.set_aborted(result)
        else:
            rospy.loginfo("POMDP planning")
            self._is_planning = True
            action = self._planner.plan(self.agent)
            if hasattr(self.agent, "tree") and self.agent.tree is not None:
                _dd = pomdp_py.utils.TreeDebugger(self.agent.tree)
                _dd.p(1)
            rospy.loginfo(f"Planning successful. Action: {action}")
            rospy.loginfo("Action published")
            action_msg = self._action_executor_class.action_to_ros_msg(
                self.agent, action, goal.goal_id)
            self._last_action_msg = action_msg
            self._last_action = action
            self._action_publisher.publish(action_msg)
            result.status = GoalStatus(status=GoalStatus.SUCCEEDED,
                                       text="planning succeeded. Action published.")
            self._plan_server.set_succeeded(result)
            self._is_planning = False

    def _action_exec_status_cb(self, status):
        if self._last_action_msg is None:
            return
        last_action_id = ActionExecutor.action_id(self._last_action_msg)
        if last_action_id != status.goal_id.id:
            rospy.logerr("action status is not for most recently planned action;"\
                         f"{last_action_id} != {status.goal_id.id}")
            return
        self._last_action_status = status
        rospy.loginfo("received action execution status")
        if status.status == GoalStatus.ACTIVE:
            rospy.loginfo(f"action {self.last_action} is active: {status.text}")
        else:
            # action execution finished. Whether it is successful, we
            # need to update the agent belief and the planner.
            rospy.loginfo(f"action {self.last_action} is finished: {status.text}")
            sources = self._observation_interpretor_class.SOURCES_FOR_REGULAR_UPDATE
            rospy.loginfo(f"collecting observations from {sources}")
            observation = self.collect_observation(sources)
            rospy.loginfo(f"observations collected")
            self.agent.update_belief(observation, self.last_action)
            rospy.loginfo("updated belief (action finished). Robot state: {}".format(self.agent.belief.mpe().s(self.agent.robot_id)))
            if hasattr(self.agent, "tree"):
                self.planner.update(self.agent, self._last_action, observation)
                print("############### action exec status planner update")
            rospy.loginfo(f"updated planner")
            self._clear_last_action()
            rospy.loginfo(typ.bold(f"CLEARED LAST ACTION."))
            self._action_publisher.publish(KeyValAction(type="nothing", stamp=rospy.Time.now()))

    def _clear_last_action(self):
        self._last_action = None
        self._last_action_msg = None
        self._last_action_status = None

    def _setup_troubleshooting_message(self):
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

    def _setup_info_message(self):
        message = f"{self.__class__.__name__} subscribes to the following observation types:\n"
        for z_type in self._observation_topics:
            message += f"- {self._observation_topics[z_type]}: {self._observation_msg_types[z_type].__name__}\n"
        return message

    def check_if_ready(self):
        raise NotImplementedError

    def belief_to_ros_msg(self, belief):
        raise NotImplementedError

    def current_progress_msg(self):
        """returns a std_msgs/String message to report task progress"""
        raise NotImplementedError

    @property
    def last_action(self):
        return self._last_action

    @property
    def last_action_status(self):
        return self._last_action_status

    def collect_observation(self, sources):
        """First, wait for an observation from each observation source
        that should be collected (determined by observation interpreter);
        Then, merge those observations into a single observation object
        and return."""
        observation_messages = ros_utils.WaitForMessages(
            [self._observation_topics[s] for s in sources],
            [self._observation_msg_types[s] for s in sources],
            queue_size=10,
            delay=0.5,
            sleep=0.05,
            verbose=True).messages
        return self._observation_interpretor_class.merge_observation_msgs(
            observation_messages, self)


class ActionExecutor:
    """ActionExecutor is meant to be run as a node by itself,
    which subscribes to the ~action topic that the BaseAgentROSBridge
    publishes when one planning step is performed.

    It:
    - subscribes to actions published at a topic, by BaseAgentROSBridge
    - executes a received action;
    - publishes status as the robot executes.

    It takes care of converting POMDP actions to the appropriate format for the
    specific robot.

    Important functions to implement:
    - action_to_ros_msg (static): converts a POMDP action to a ROS message
    - execute_action_cb: called when an action message is received, and
        execute that action on the robot.
    """
    def __init__(self,
                 action_topic="~action", status_topic="~status"):
        self.node_name = rospy.get_name()
        self._action_topic = action_topic  # The topic to subscribe to to receive actions
        self._status_topic = status_topic  # The topic to publish status
        self._action_msg_type = KeyValAction

    @property
    def status_topic(self):
        return "{}/{}".format(self.node_name, self._status_topic)

    def setup(self):
        self._status_pub = rospy.Publisher(self._status_topic,
                                           GoalStatus,
                                           queue_size=10, latch=True)
        self._action_sub = rospy.Subscriber(self._action_topic,
                                            self._action_msg_type,
                                            self._execute_action_cb)

    def _execute_action_cb(self, action_msg):
        """Handles action execution"""
        raise NotImplementedError

    @classmethod
    def action_id(cls, action_msg):
        assert isinstance(action_msg, KeyValAction),\
            f"action_msg should be KeyValAction, but got {type(action_msg)}"
        return "{}-{}".format(action_msg.type, str(action_msg.stamp))

    @classmethod
    def action_to_ros_msg(cls, agent, action, goal_id):
        """
        Given a POMDP agent and an action for that POMDP,
        output a ROS message. (robot-specific)

        Args:
            agent (pomdp_py.Agent)
            action (pomdp_py.Action)
        """
        raise NotImplementedError

    def publish_status(self, status, text, action_id, stamp):
        status = GoalStatus(status=status,
                            text=text)
        status.goal_id.id = action_id
        status.goal_id.stamp = stamp
        if status == GoalStatus.ABORTED or status == GoalStatus.REJECTED:
            rospy.logerr(text)
        else:
            rospy.loginfo(text)
        self._status_pub.publish(status)


class ObservationInterpreter:
    """The observation interpreter provides the callback function for an observation type.
       Its job is to interpret an observation and update the agent's belief. It
       also takes care of publishing the agent's belief in a suitable format for
       visualization.
    """
    # Should map from observation type (class) to a callback function
    # To be filled by child class. The callback may affect the agent's belief.
    CALLBACKS = {}

    # observation types that will be collected once an action
    # is completed and a round of planner and belief update is performed.
    SOURCES_FOR_REGULAR_UPDATE = []

    SKIP_WARNING_PRINTED = {}

    @classmethod
    def get_observation_callback(cls, z_type):
        def dummy_cb(z_msg, args):
            if not cls.SKIP_WARNING_PRINTED.get(z_type):
                rospy.logwarn("skipping observation ({})".format(type(z_msg)))
                cls.SKIP_WARNING_PRINTED[z_type] = True

        if z_type not in cls.CALLBACKS:
            rospy.logwarn(f"Observation interpreter does not handle observation type {z_type}")
            return dummy_cb
        else:
            return cls.CALLBACKS[z_type]

    @classmethod
    def merge_observation_msgs(cls, observation_msgs, bridge):
        """
        Given multiple observation messages, return a single Observation object
        (useful for object-oriented POMDPs, where different observation messages
        are observations for different objects, while the belief update happens
        given a single POMDP observation object that is the joint of those observations.).
        """
        raise NotImplementedError
