# Observation interpretation specific to Spot
import rospy
from geometry_msgs.msg import PoseStamped
from sloop_ros.msg import GridMap2d
from sloop_object_search.ros.framework import ObservationInterpreter
from sloop_object_search.ros.grid_map_utils import ros_msg_to_grid_map
from sloop_object_search.ros.mapinfo_utils import FILEPATHS, register_map
from sloop_object_search.ros.sloop_mos import (grid_map_msg_callback,
                                               robot_pose_msg_callback)
from sloop_object_search.ros.ros_utils import tf2_transform
from sloop_object_search.oopomdp.domain.observation import ObjectDetection2D, GMOSObservation

import tf2_ros


def detection_3d_msg_callback(detection_msg, bridge):
    """For object detections which have depth readings.
    Args:
        detection_msg (rbd_spot_perception/SimpleDetection3DArray)
    """
    if bridge.agent is None:
        return

    # For each detection, if the detected class is acceptable,
    # then obtain the center of the 3D pose, project it down
    # to the grid map. And use this observation to update agent belief.
    detectable_objects = bridge.agent.agent_config["detectable_objects"]
    z_joint_dict = {label: ObjectDetection2D(label, ObjectDetection2D.NULL)
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
            zobj = ObjectDetection2D(det3d.label, (obj_grid_x, obj_grid_y))
            z_joint_dict[det3d.label] = zobj
            rospy.loginfo("- {} ({:.3f}) grid loc: {}".format(det3d.label, det3d.score, zobj.loc))
    z_joint = GMOSObservation(z_joint_dict)
    bridge.agent.update_belief(z_joint, bridge.last_action_executed)
    rospy.loginfo("updated belief")
    bridge.visualize_current_belief()
    rospy.loginfo("updated visualization")


def detection_img_msg_callback(detection_msg, bridge):
    """For object detections in image, but we don't have depth"""


class SpotObservationInterpreter(ObservationInterpreter):
    CALLBACKS = {"grid_map": grid_map_msg_callback,
                 "robot_pose": robot_pose_msg_callback,
                 "detection_3d": detection_3d_msg_callback}
