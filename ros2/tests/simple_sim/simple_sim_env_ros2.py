#!/usr/bin/env python
#
# Example command to run:
# ros2 run genmos_object_search_ros2 simple_sim_env_ros2.py config_simple_sim_lab121_lidar.yaml
# ros2 launch simple_sim_env_ros2.launch map_name:=lab121_lidar
import math
import time
import numpy as np
import copy
import argparse
import yaml
import pomdp_py

import rclpy
from rclpy.node import Node

from genmos_ros2 import ros2_utils


class SimpleSimEnv(pomdp_py.Environment):
    """This is a simple 3D environment. All of its coordinates should
    be in the world frame, as this is meant to support the underlying
    simulation of an object search scenario in ROS 2."""
    def __init__(self, env_config, objloc_index=None):
        # Get initial robot pose
        self.env_config = env_config

        # agent config is the same format as what is given to a SLOOP MosAgent creation.
        self.agent_config = env_config["agent_config"]
        self._robot_pose_topic = "~init_robot_pose"
        robot_pose_msg = ros_utils.WaitForMessages([self._robot_pose_topic], [PoseStamped], verbose=True)\
                                  .messages[0]
        init_robot_pose = ros_utils.pose_tuple_from_pose_stamped(robot_pose_msg)

        self.robot_id = self.agent_config["robot"].get("id", "robot")
        init_robot_state = RobotState(self.robot_id,
                                      init_robot_pose, (), None)

        objects = self.agent_config["objects"]
        object_locations = self.env_config["object_locations"]
        object_states = {}

        # object locations should be a list of 3D locations;
        # there should be an equal number of locations for every object,
        # indicating different environment configurations.
        N = -1
        for objid in object_locations:
            assert type(object_locations[objid]) == list\
                and type(object_locations[objid][0]) == list,\
                "object locations should be a list of 3D locationS"
            if N == -1:
                N = len(object_locations[objid])
            else:
                assert N == len(object_locations[objid]),\
                    "there should be an equal number of location configs for every object"
        self._num_objloc_configs = N
        self._objloc_index = objloc_index if objloc_index is not None else 0
        for objid in objects:
            objloc = object_locations[objid][self._objloc_index]
            sobj = ObjectState(objid, objects[objid]["class"], tuple(objloc))
            object_states[objid] = sobj
        init_state = pomdp_py.OOState({self.robot_id: init_robot_state,
                                       **object_states})

        # Transition model; Note that the agent will have a 3D
        # sensor - because it operates in 3D. So if the agent_config's
        # sensor is 2D, we will convert it.
        _dms = init_detection_models(self.agent_config)
        detection_models = {}
        for objid in _dms:
            detection_models[objid] = ensure_detection_model_3d(_dms[objid])
        self.detection_models = detection_models
        self.no_look = self.agent_config["robot"].get("no_look", True)
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look, pos_precision=0.0001, rot_precision=0.0001)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # Also make an observation model
        robot = self.agent_config["robot"]
        self.localization_model = interpret_localization_model(robot)
        self.robot_observation_model = RobotObservationModel(
            robot['id'], localization_model=self.localization_model)
        self.observation_model = GMOSObservationModel(
            robot["id"], self.detection_models,
            robot_observation_model=self.robot_observation_model,
            no_look=self.no_look)

        # Reward model
        target_ids = self.agent_config["targets"]
        reward_model = GoalBasedRewardModel(target_ids, robot_id=self.robot_id)
        super().__init__(init_state,
                         transition_model,
                         reward_model)

    def bump_objloc_index(self):
        self._objloc_index = (self._objloc_index + 1) % self._num_objloc_configs
        return self._objloc_index

    def reset_objloc_index(self):
        self._objloc_index = 0
        return self._objloc_index

    def reachable(self, pos):
        return True  # the real world has no bound

    def provide_observation(self, action=None):
        # will use own observation model.
        observation = self.observation_model.sample(self.state, action)

        # convert voxels into object detections -- what a robot would receive
        real_zobjs = {self.robot_id: observation.z(self.robot_id)}
        for objid in observation:
            if objid == self.robot_id:
                continue
            if isinstance(observation.z(objid), ObjectVoxel):
                voxel = observation.z(objid)
                if voxel.label == objid:
                    objsizes = self.object_spec(objid).get("sizes", [0.12, 0.12, 0.12])
                    zobj = ObjectDetection(objid, voxel.loc, sizes=objsizes)
                else:
                    zobj = ObjectDetection(objid, ObjectDetection.NULL)
                real_zobjs[objid] = zobj

        return GMOSObservation(real_zobjs)

    def object_spec(self, objid):
        return self.agent_config["objects"][objid]

    def get_2d_state(self):
        robot_pose = self.state.s(self.robot_id).pose
        x, y, _ = robot_pose[:3]
        _, _, yaw = math_utils.quat_to_euler(*robot_pose[3:])
        srobot2d = RobotState(self.robot_id,
                              (x, y, yaw),
                              self.state.s(self.robot_id).objects_found,
                              self.state.s(self.robot_id).camera_direction)
        sobjs = {self.robot_id: srobot2d}
        for objid in self.state.object_states:
            if objid == self.robot_id:
                continue
            sobj = self.state.s(objid)
            x, y = sobj.loc[:2]
            sobj2d = ObjectState(objid,
                                 sobj.objclass,
                                 (x,y))
            sobjs[objid] = sobj2d
        return pomdp_py.OOState(sobjs)


class SimpleSimEnvROSNode(ros2_utils.WrappedNode):
    """
    note that all messages that this node receives
    should be in the world frame.

    Subscribes:
      ~pomdp_action (KeyValAction)
      ~reset (String)
    Publishes:
      ~robot_pose (PoseStamped)
      ~state_markers (MarkerArray)
      ~pomdp_observation (KeyValObservation)
    """
    NODE_NAME="simple_sim_env"
    def __init__(self, config, verbose=True):
        super().__init__(SimpleSimEnvROSNode.NODE_NAME,
                         params=[("state_pub_rate", 10),
                                 ("observation_pub_rate", 3),
                                 ("world_frame", "map")],
                         verbose=verbose)
        self._init_config = config

        state_pub_rate = self.get_parameter("state_pub_rate")
        observation_pub_rate = self.get_parameter("observation_pub_rate")
        self.world_frame = self.get_parameter("world_frame")  # fixed frame of the world

        # self.br = TransformBroadcaster()

        action_topic = "~/pomdp_action"
        reset_topic = "~/reset"
        state_markers_topic = "~/state_markers"
        robot_pose_topic = "~/robot_pose"
        observation_topic = "~/pomdp_observation"
        # self.env = SimpleSimEnv(config)
        # self.action_sub = rospy.Subscriber(action_topic, KeyValAction, self._action_cb)
        # self.reset_sub = rospy.Subscriber(reset_topic, String, self._reset_cb)

        # self.state_markers_pub = rospy.Publisher(state_markers_topic, MarkerArray, queue_size=10)
        # self.state_pub_rate = state_pub_rate

        # self.robot_pose_pub = rospy.Publisher(robot_pose_topic, PoseStamped, queue_size=10)

        # # observation
        # self.observation_pub = rospy.Publisher(observation_topic, KeyValObservation, queue_size=10)
        # self.observation_pub_rate = observation_pub_rate

        # # navigation related
        # self.nav_step_duration = rospy.get_param("~step_duration", 0.1)  # amount of time to execute one step
        # self.translation_step_size = rospy.get_param("~translation_step_size", 0.1)  # in meters
        # self.rotation_step_size = rospy.get_param("~rotation_step_size", 5)  # in degreesw
        # assert self.translation_step_size > 0, "translation_step_size must be > 0"
        # assert self.rotation_step_size > 0, "rotation_step_size must be > 0"
        # self._navigating = False
        # self._action_done_pub = rospy.Publisher("~action_done", String, queue_size=10, latch=True)  # publishes when action is done



def main():
    parser = argparse.ArgumentParser(
        description="Starts the SimpleSim environment")
    parser.add_argument("config_file", type=str,
                        help="path to YAML configuration file"\
                             "(plain dictionary, non-ROS2 format)")
    args, node_cmd_args = parser.parse_known_args()
    if len(node_cmd_args) == 0:
        node_cmd_args = None

    rclpy.init(args=node_cmd_args)
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    n = SimpleSimEnvROSNode(config)
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# import rclpy
# from rclpy.node import Node
# def main(args=None):
#     rclpy.init(args=args)
#     node = Node('my_node_name')
#     rclpy.spin(node)
#     rclpy.shutdown()
# if __name__ == '__main__':
#     main()
