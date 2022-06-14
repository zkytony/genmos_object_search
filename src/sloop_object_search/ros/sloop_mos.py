import rospy
from .framework import BaseAgentWrapper
import sloop_ros.msg as sloop_ros
from sloop_object_search.oopomdp.agent import make_agent
from sloop_object_search.oopomdp.planner import make_planner

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


class SloopMosROSAgentWrapper(BaseAgentWrapper):
    def __init__(self, oopomdp_agent, planner, ros_config={}):
        self.planner = planner
        super().__init__(oopomdp_agent,
                         ros_config=ros_config,
                         planner=planner)
