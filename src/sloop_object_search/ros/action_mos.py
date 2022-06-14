import sloop_ros.msg as sloop_ros
from sloop_object_search.oopomdp.domain.action import (MotionAction2D,
                                                       MotionActionTopo,
                                                       LookAction,
                                                       FindAction,
                                                       StayAction)

def action_to_ros_msg(action, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()

    if isinstance(action, MotionAction2D):
        sloop_ros.SloopMosAction(stamp=stamp,
                                 type="motion_2d",
                                 fields=["motion", "step_cost", "motion_name"],
                                 values=[str(action.motion), str(action.step_cost), str(action.motion_name)])
    elif isinstance(action, StayAction):
        sloop_ros.SloopMosAction(stamp=stamp,
                                 type="stay",
                                 fields=["nid", "cost_scaling_factor"],
                                 values=[action.src_nid, action._cost_scaling_factor])
    elif isinstance(action, MotionActionTopo):
        sloop_ros.SloopMosAction(stamp=stamp,
                                 type="motion_topo",
                                 fields=["src_nid", "dst_nid", "gdist", "cost_scaling_factor"],
                                 values=[action.src_nid, action.dst_nid, action.gdist, action._cost_scaling_factor])
    elif isinstance(action, LookAction):
        sloop_ros.SloopMosAction(stamp=stamp,
                                 type="look",
                                 fields=[],
                                 values=[])
    elif isinstance(action, FindAction):
        sloop_ros.SloopMosAction(stamp=stamp,
                                 type="find",
                                 fields=[],
                                 values=[])
