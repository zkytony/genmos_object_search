import os
import rospy
import sloop_ros.msg as sloop_ros
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
from tf.transformations import euler_from_quaternion
from sloop_object_search.oopomdp.agent import make_agent as make_sloop_mos_agent
from sloop_object_search.oopomdp.planner import make_planner as make_sloop_mos_planner
from sloop_object_search.utils.misc import import_class
from sloop_object_search.utils.math import to_degrees
from sloop_object_search.oopomdp.agent.visual import visualize_step
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_ros.msg import GridMap2d
## For ROS-related programs, we should import FILEPATHS and MapInfoDataset this way.
from .mapinfo_utils import FILEPATHS, MapInfoDataset, register_map
from .framework import BaseAgentROSRunner
from .action_mos import action_to_ros_msg


class SloopMosAgentROSRunner(BaseAgentROSRunner):
    """
    Interfaces between SLOOP MOS Agent and ROS.
    """
    def __init__(self, ros_config={}):
        super().__init__(ros_config=ros_config)
        self.mapinfo = MapInfoDataset()
        self.viz = None
        self._grid_map = None
        self._init_robot_pose = None

    def check_if_ready(self):
        # Check if I have grid map and I have the robot pose
        return self._grid_map is not None and self._init_robot_pose is not None

    def _interpret_grid_map_msg(self, grid_map_msg):
        if self._grid_map is not None:
            return

        grid_map = ros_msg_to_grid_map(grid_map_msg)

        # If grid map's name is unrecognized, we would like
        # to register this map into our database.
        if grid_map.name not in FILEPATHS:
            register_map(grid_map)

        self.mapinfo.load_by_name(grid_map.name)
        self._grid_map = grid_map
        rospy.loginfo("Obtained grid map")

    def _interpret_robot_pose_msg(self, robot_pose_msg):
        if self._init_robot_pose is not None:
            return

        if self._grid_map is None:
            # We can't interpret robot pose yet
            return

        # first, obtain the position in grid map coords
        metric_position = robot_pose_msg.pose.position
        quat = robot_pose_msg.pose.orientation
        rx, ry = self._grid_map.to_grid_pos(metric_position.x, metric_position.y)
        _, _, yaw = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
        yaw = to_degrees(yaw)
        self._init_robot_pose = (rx, ry, yaw)
        rospy.loginfo(f"initial robot pose (on grid map) set to {self._init_robot_pose}")

    def _observation_cb(self, observation_msg):
        """
        we can deal with three types of observations:
        - robot state
        - object detection
        - grid map
        - spatial language
        """
        if isinstance(observation_msg, GridMap2d):
            self._interpret_grid_map_msg(observation_msg)
        elif isinstance(observation_msg, geometry_msgs.PoseStamped):
            self._interpret_robot_pose_msg(observation_msg)
        else:
            rospy.logerr(f"Does not know how to handle observation of type {type(observation_msg)}")

    def run(self):
        # start visualization
        _config = self.agent.agent_config
        self.viz = import_class(_config["visualizer"])(self.agent.grid_map,
                                                       bg_path=FILEPATHS[self.agent.map_name]["map_png"],
                                                       **_config["viz_params"])
        self._visualize_step()
        super().run()

    def make_agent(self, config):
        return make_sloop_mos_agent(config, init_pose=self._init_robot_pose)

    def make_planner(self, config, agent):
        return make_sloop_mos_planner(config["planner_config"], agent)

    def _visualize_step(self, action=None, **kwargs):
        colors = {j: self.agent.agent_config["objects"][j].get("color", [128, 128, 128])
                  for j in self.agent.belief.object_beliefs
                  if j != self.agent.robot_id}
        no_look = self.agent.agent_config["no_look"]
        draw_fov = list(self.agent.belief.object_beliefs.keys())
        if not no_look:
            if not isinstance(action, LookAction):
                draw_fov = None
        if action is None:
            draw_fov = None
        self.viz.visualize(self.agent, {}, colors=colors, draw_fov=draw_fov, **kwargs)

    def belief_to_ros_msg(self, belief, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()

        bobj_msgs = []
        for objid in belief.object_beliefs:
            bobj = belief.object_beliefs[objid]
            locations = []
            probs = []
            objclass = None
            for sobj in bobj:
                locations.append(sloop_ros.Loc(x=sobj.loc[0], y=sobj.loc[1]))
                probs.append(bobj[sobj])
                if objclass is None:
                    objclass = sobj.objclass
            bobj_msg = sloop_ros.SloopMosObjectBelief(stamp=stamp, objid=objid, objclass=objclass)
            bobj_msgs.append(bobj_msg)
        belief_msg = sloop_ros.SloopMosBelief(stamp=stamp, object_beliefs=bobj_msgs)
        return belief_msg

    def action_to_ros_msg(self, action, stamp=None):
        return action_to_ros_msg(action, stamp=stamp)
