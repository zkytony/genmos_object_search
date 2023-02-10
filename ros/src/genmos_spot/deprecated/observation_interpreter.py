# Observation interpretation specific to Spot
import rospy
import sloop.observation
from pomdp_py.utils import typ
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatus
from genmos_object_search_ros.msg import GridMap2d

from genmos_object_search.utils.colors import lighter, inverse_color_rgb
from genmos_object_search.oopomdp.domain.observation import (
    ObjectDetection, GMOSObservation, RobotObservationTopo)
from genmos_object_search.oopomdp.domain.action import FindAction
from genmos_object_search.oopomdp.models.detection_models import FanModelSimpleFPLabelOnly
from genmos_ros.framework import ObservationInterpreter
from genmos_ros.grid_map_utils import ros_msg_to_grid_map
from genmos_ros.mapinfo_utils import FILEPATHS, register_map
from genmos_ros.genmos import (grid_map_msg_callback,
                                     robot_pose_msg_callback,
                                     interpret_grid_map_msg,
                                     interpret_robot_pose_msg)
from genmos_ros.ros_utils import tf2_transform

import tf2_ros

###### The observation interpetation functions and callbacks #########
def interpret_detection_3d_msg(detection_msg, bridge):
    if bridge.agent is None:
        return

    # For each detection, if the detected class is acceptable,
    # then obtain the center of the 3D pose, project it down
    # to the grid map. And use this observation to update agent belief.
    detectable_objects = bridge.agent.agent_config["detectable_objects"]
    z_joint_dict = {label: ObjectDetection(label, ObjectDetection.NULL)
                    for label in detectable_objects}

    rospy.loginfo("detected objects (3D):")
    for det3d in detection_msg.detections:
        if det3d.label in detectable_objects:
            # Get detection position in map frame
            det_pose_stamped = PoseStamped(header=detection_msg.header, pose=det3d.box.center)
            det_pose_stamped_map_frame =\
                tf2_transform(bridge.tfbuffer, det_pose_stamped, bridge.map_frame)
            if det_pose_stamped_map_frame is None:
                # unable to get pose in map frame
                return

            # Map detection position to grid map position and build observation
            obj_metric_position = det_pose_stamped_map_frame.pose.position
            obj_grid_x, obj_grid_y = bridge.agent.grid_map.to_grid_pos(
                obj_metric_position.x, obj_metric_position.y)
            zobj = ObjectDetection(det3d.label, (obj_grid_x, obj_grid_y))
            z_joint_dict[det3d.label] = zobj
            rospy.loginfo("- {} ({:.3f}) grid loc: {}".format(det3d.label, det3d.score, zobj.loc))
            # # highlight the object detection location on the map
            # bridge.add_visual_highlights([(obj_grid_x, obj_grid_y)],
            #                              inverse_color_rgb(bridge.object_color(det3d.label)),
            #                              shape="circle", alpha=0.4, scale=0.8)
    return z_joint_dict


def detection_3d_msg_callback(detection_msg, bridge):
    """For object detections which have depth readings.
    Args:
        detection_msg (rbd_spot_perception/SimpleDetection3Array)
    """
    z_joint_dict = interpret_detection_3d_msg(detection_msg, bridge)
    if z_joint_dict is None:
        return z_joint_dict

    z_joint = GMOSObservation(z_joint_dict)
    bridge.agent.update_belief(z_joint, bridge.last_action)
    rospy.loginfo("updated belief. Robot state: {}".format(bridge.agent.belief.mpe().s(bridge.agent.robot_id)))

def detection_img_msg_callback(detection_msg, bridge):
    """For object detections in image, but we don't have depth"""


def spatial_lang_callback(splang_msg, bridge):
    splang = splang_msg.data
    map_name = bridge.agent.map_name
    splang_obz = sloop.observation.parse(splang,
                                         map_name,
                                         kwfile=FILEPATHS[map_name]["symbol_to_synonyms"],
                                         spacy_model=bridge.agent.spacy_model,
                                         verbose_level=1)
    bridge.agent.update_belief(splang_obz, None)
    rospy.loginfo(typ.bold("updated belief based on spatial language."))


###### The actual ObservationInterpreter #########
class SpotObservationInterpreter(ObservationInterpreter):
    CALLBACKS = {"grid_map": grid_map_msg_callback,
                 "robot_pose": robot_pose_msg_callback,
                 "detection_3d": detection_3d_msg_callback,
                 "spatial_lang": spatial_lang_callback}

    # observation types that will be collected once an action
    # is completed and a round of planner and belief update is
    # performed.
    SOURCES_FOR_REGULAR_UPDATE = ["robot_pose", "detection_3d"]

    @classmethod
    def merge_observation_msgs(cls, observation_msgs, bridge):
        robot_pose_msg = observation_msgs[0]
        robot_pose = interpret_robot_pose_msg(robot_pose_msg, bridge)
        detection_3d_msg = observation_msgs[1]
        z_joint_dict = interpret_detection_3d_msg(detection_3d_msg, bridge)

        # Note that this function is called when the planner is updated.
        # If the robot's observation model doesn't want, the search tree
        # would have object detection observations with "NO_POSE" as the location.
        # We want to match that.
        for objid in bridge.agent.observation_model.detection_models:
            detection_model = bridge.agent.observation_model.detection_models[objid]
            if isinstance(detection_model, FanModelSimpleFPLabelOnly):
                z_obj = z_joint_dict[objid]
                if z_obj.loc is not None:
                    z_joint_dict[objid] = ObjectDetection(objid, ObjectDetection.NO_POSE)

        # Now, we need to figure out the robot observation, which
        # involves figuring out whether an object is found, and what
        # the new topo nid is.

        # We want this function to be called only when the last action is finished
        assert not(bridge.last_action_status is None\
                   or bridge.last_action_status == GoalStatus.ACTIVE)

        # Now, if the last action is Find, then check if any of the target
        # object is in the detection_3d_msg. If so, then the object is
        # considered found.
        newly_found_objects = set()
        if isinstance(bridge.last_action, FindAction):
            for det3d in detection_3d_msg.detections:
                if det3d.label in bridge.agent.agent_config["targets"]:
                    newly_found_objects.add(det3d.label)
        current_robot_state = bridge.agent.belief.mpe().s(bridge.agent.robot_id)
        objects_found = tuple(sorted(set(current_robot_state["objects_found"]) | newly_found_objects))

        # If the last action is MoveTopo, then just update the topo nid to
        # be the closest node the robot is at now (doesn't matter if it's not
        # what the action wants as destination);
        topo_nid = bridge.agent.topo_map.closest_node(*robot_pose[:2])

        robot_observation = RobotObservationTopo(
            bridge.agent.robot_id,
            robot_pose,
            objects_found,
            None,
            topo_nid)
        z_joint_dict[bridge.agent.robot_id] = robot_observation
        return GMOSObservation(z_joint_dict)
