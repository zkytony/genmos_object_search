import rospy
import sloop_ros.msg as sloop_ros
import std_msgs.msg as std_msgs
from sloop_object_search.oopomdp.agent import make_agent
from sloop_object_search.oopomdp.planner import make_planner
from sloop_object_search.utils.misc import import_class
from sloop_object_search.oopomdp.agent.visual import visualize_step
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_ros.msg import GridMap2d
from sloop.osm.datasets import FILEPATHS
from .framework import BaseAgentROSInterface
from .action_mos import action_to_ros_msg

class SloopMosAgentROSInterface(BaseAgentROSInterface):
    def __init__(self, planner, ros_config={}):
        self.planner = planner
        self.viz = None
        super().__init__(ros_config=ros_config,
                         planner=planner)

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

    def _observation_cb(self, observation_msg):
        """
        we can deal with three types of observations:
        - robot state
        - object detection
        - grid map
        - spatial language
        """
        if isinstance(observation_msg, GridMap2d):
            self._interpret_grid_map_msg(observation_msg):
        raise NotImplementedError

    def _interpret_grid_map_msg(self, grid_map_msg):
        grid_map = ros_msg_to_grid_map(grid_map_msg)
        pass

    def interpret_observation_msg(self, belief, stamp=None):
        pass

    def run(self):
        # start visualization
        _config = self.agent.agent_config
        self.viz = import_class(_config["visualizer"])(self.agent.grid_map,
                                                       bg_path=FILEPATHS[self.agent.map_name]["map_png"],
                                                       **_config["viz_params"])
        self._visualize_step()
        super().run()

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

def make_ros_agent(_config):
    """
    Args:
        config (dict): the configuration dictionary; refer to examples
            under sloop_ros/tests
    """
    _ros_config = _config.get("ros_config", {})
    init_pose_topic = _ros_config.get("init_pose_topic", "~init_pose")
    rospy.loginfo(f"Waiting for initial pose at topic: {init_pose_topic}")
    init_pose_msg = sloop_ros.GridMapPose2d(x=5,y=5,yaw=0.0)
    # rospy.wait_for_message(init_pose_topic,
    #                                        sloop_ros.GridMapPose2d)
    agent = make_agent(_config, init_pose=(init_pose_msg.x, init_pose_msg.y, init_pose_msg.yaw))
    planner = make_planner(_config["planner_config"], agent)
    return SloopMosROSAgentWrapper(agent, planner,
                                   ros_config=_config.get("ros_config", {}))
