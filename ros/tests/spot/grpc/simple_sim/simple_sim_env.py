#!/usr/bin/env python
# To run this, use the 'simple_sim_env.launch' file:
#    roslaunch sloop_object_search_ros simple_sim_env.launch map_name:=<map_name>
# Also run:
#    roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=<map_name>

import rospy
import pomdp_py

from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point, Quaternion, Vector3, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sloop_object_search_ros.msg import KeyValAction, KeyValObservation

from sloop_mos_ros import ros_utils
from sloop_object_search.oopomdp.domain.state import ObjectState, RobotState
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from sloop_object_search.oopomdp.agent.common import (init_object_transition_models,
                                                      init_detection_models)
from sloop_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from sloop_object_search.utils.misc import hash16


class SimpleSimEnv(pomdp_py.Environment):
    """This is a simple 3D environment. All of its coordinates should
    be in the world frame, as this is meant to support the underlying
    simulation of an object search scenario in ROS."""
    def __init__(self, env_config):
        # Get initial robot pose
        self.env_config = env_config
        self._robot_pose_topic = "~init_robot_pose"
        robot_pose_msg = ros_utils.WaitForMessages([self._robot_pose_topic], [PoseStamped], verbose=True)\
                                  .messages[0]
        init_robot_pose = ros_utils.pose_tuple_from_pose_stamped(robot_pose_msg)
        self.robot_id = self.env_config.get("robot_id", "robot")
        init_robot_state = RobotState(self.robot_id,
                                      init_robot_pose, (), None)

        assert "objects" in self.env_config, "env_config needs 'objects'"
        objects = self.env_config["objects"]
        object_states = {}
        for objid in objects:
            sobj = ObjectState(objid, objects[objid]["class"], objects[objid]["pos"])
            object_states[objid] = sobj
        init_state = pomdp_py.OOState({self.robot_id: init_robot_state,
                                       **object_states})

        # Transition model
        assert "robot" in self.env_config and "detectors" in self.env_config["robot"],\
            "env_config needs 'robot', which needs 'detectors'."
        self.detection_models = init_detection_models(self.env_config)
        self.no_look = env_config["robot"].get("no_look", True)
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.env_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # Reward model
        target_ids = self.env_config["targets"]
        reward_model = GoalBasedRewardModel(target_ids, robot_id=self.robot_id)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    def reachable(self, pos):
        return True  # the real world has no bound

    def provide_observation(self, observation_model, action):
        pass

    def object_spec(self, objid):
        return self.env_config["objects"][objid]


class SimpleSimEnvROSNode:
    """
    note that all messages that this node receives or publishes
    should be in the world frame.

    Subscribes:
      ~action (KeyValAction)
    Publishes:
      ~state_markers (MarkerArray)
      ~observation (KeyValObservation)
    """
    def __init__(self, node_name="simple_sim_env"):
        self.name = "simple_sim_env"
        rospy.init_node(self.name)
        config = rospy.get_param("~config")  # access parameters together as a dictionary
        state_pub_rate = rospy.get_param("~state_pub_rate", 10)
        self.world_frame = rospy.get_param("~world_frame")  # fixed frame of the world

        action_topic = f"~pomdp_action"
        state_markers_topic = f"~state_markers"
        self.env = SimpleSimEnv(config)
        self.action_sub = rospy.Subscriber(action_topic, KeyValAction, self._action_cb)

        self.state_markers_pub = rospy.Publisher(
            state_markers_topic, MarkerArray, queue_size=10)
        self.state_pub_rate = rospy.Rate(state_pub_rate)

    def run(self):
        rospy.loginfo("publishing state markers")
        while not rospy.is_shutdown():
            state_markers_msg = self._make_state_markers(self.env.state)
            self.state_markers_pub.publish(state_markers_msg)
            self.state_pub_rate.sleep()

    def _action_cb(self, action_msg):
        if action_msg.type == "move":
            pass

    def _make_state_markers(self, state):
        markers = []
        header = Header(stamp=rospy.Time.now(),
                        frame_id=self.world_frame)
        for objid in state.object_states:
            if objid == self.env.robot_id:
                continue  # we draw the robot separately
            viz_type_name = self.env.object_spec(objid).get("viz_type", "cube")
            viz_type = eval(f"Marker.{viz_type_name.upper()}")
            color = self.env.object_spec(objid).get("color", [0.0, 0.8, 0.0, 0.8])
            obj_marker = ros_utils.make_viz_marker_from_object_state(
                state.s(objid), header, viz_type=viz_type,
                color=color, scale=0.5, lifetime=1.0)
            markers.append(obj_marker)

        # TODO: consider use arrow
        robot_marker = ros_utils.make_viz_marker_from_robot_state(
                state.s(self.env.robot_id), header, viz_type=Marker.CUBE,
                color=[0.9, 0.1, 0.1, 0.8], scale=0.75, lifetime=1.0)
        markers.append(robot_marker)
        return MarkerArray(markers)

def main():
    n = SimpleSimEnvROSNode()
    n.run()

if __name__ == "__main__":
    main()
