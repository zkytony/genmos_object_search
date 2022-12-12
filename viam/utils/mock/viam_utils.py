# Functions in this file have the same name as
# those under utils/viam_utils, but do not require
# connection to viam server or any rpc to return.

from sloop_object_search.grpc import observation_pb2 as o_pb2
from sloop_object_search.grpc import common_pb2
from sloop_object_search.grpc.utils import proto_utils
from sloop_object_search.utils import math as math_utils

from ..viam_utils import Quaternion, OrientationVector, AxisAngle

import numpy as np


MOCK_ROBOT_POSE = None
MOCK_TARGET_LOC = [0.05, 1.37, 0.5]

async def connect_viamlab_ur5():
    return None

async def viam_get_ee_pose(viam_robot, arm_name="arm"):
    """return current end-effector pose in world
    frame through Viam.
    Return type: tuple (x,y,z,qx,qy,qz,qw)
    Note that viam's positions units are in milimeters.
    We will convert them into meters (more familiar with me)"""
    return MOCK_ROBOT_POSE

async def viam_get_point_cloud_array(viam_robot, target_frame="camera"):
    return None

def viam_get_object_detections3d(world_frame):
    """Return type: a list of (label, box3d) tuples.
    A label is a string.
    A box3d is a tuple (center, w, l, h)
    Note that we want 'center' in the world frame. In
    the case of a tabletop robot, it should be the frame
    of its base."""
    # NOTE: SKIPPING THIS BECAUSE WILL NOT USE POINT CLOUD AS
    # THE depth-cam is UNRELIABLE FOR THE UR5 GRIPPER AT VIAM LAB.
    raise NotImplementedError()

async def viam_get_image(viam_robot, camera_name,
                         return_type="PIL", timeout=3):
    raise NotImplementedError()


async def viam_get_object_detections2d(
        viam_robot,
        camera_name="segmenter-cam",
        detector_name="find_objects",
        confidence_thres=None,
        timeout=3):
    """
    Args:
        viam_robot_or_vision_client: either viam_robot connection, or VisionServiceClient
        camera_name (str): name of camera with color image
        detector_name (str): name of RGB object detection
        confidence_thres (float): filter out detections with confidence below this threshold
        timeout: wait time for the vision service to return a message. Must not be None.
    Returns:
        Return type: a list of (label, confidence, box2d) tuples.
        A label is a string. confidence is score, box2d is xyxy tuple
    """
    if MOCK_ROBOT_POSE is not None:
        # robot_pos3d = MOCK_ROBOT_POSE[:3]
        # # vector from robot to target
        # vec_robot_target = math_utils.vec(robot_pos3d, MOCK_TARGET_LOC)

        # # angle between robot's look vector and vec_robot_target
        # # the robot camera by default should look at +x. See sensors.py
        # default_camera_direction = np.array([1, 0, 0, 1])
        # R = math_utils.R_quat(*MOCK_ROBOT_POSE[3:], affine=True)
        # d_transformed = np.transpose(np.matmul(R, np.transpose(default_camera_direction)))
        # vec_look = d_transformed[:3]

        # angle_diff = math_utils.angle_between(vec_look, vec_robot_target)
        # if angle_diff < 30:
        return [("cup", 0.9, (-1,-1,-1,-1))]  # successful, label-only detection
        # else:
        #     return []  # no detection

    return []  # no detection


def viam_detections3d_to_proto(robot_id, detections):
    """Parameters:
    detections: a list of (label, box3d) tuples.
    A label is a string.
    A box3d is a tuple (center, w, l, h) -- this is interpreted from Viam's proto def.
    Note that 'center' should already be in world frame.
    Note: no handling of confidence.
    Note: this method is NEVER TESTED (11/06/22 12:29).
    """
    raise NotImplementedError()

def viam_detections2d_to_proto(robot_id, detections):
    """
    Args:
        detections: list of (label, confidence, xyxy) tuples.
    Returns:
        a ObjectDetectionArray proto with 2D detections.
    """
    detections_pb = []
    for det2d in detections:
        label, confidence, box2d = det2d
        x1, y1, x2, y2 = box2d
        box_pb = common_pb2.Box2D(x_min=x1, y_min=y1,
                                  x_max=x2, y_max=y2)
        det2d_pb = o_pb2.Detection(label=label,
                                   box_2d=box_pb)
        detections_pb.append(det2d_pb)
    header = proto_utils.make_header()
    return o_pb2.ObjectDetectionArray(header=header,
                                      robot_id=robot_id,
                                      detections=detections_pb)

async def viam_move(viam_robot, component_name, goal_pose, goal_frame,
                    world_state, timeout=10):
    """
    Moves the component to the given goal position and orientation.
    If move is successful, return True. Otherwise, return false.

    Args:
        viam_robot: viam grpc connection channel
        component_name (str): the name of the component we are moving
        goal_pose: a tuple (x,y,z,qx,qy,qz,qw) that specifies the
            position and orientation (quaternion) of the arm ee.
            Note that we have the position in meters. Will convert to
            milimeters for viam.
        goal_frame (str): name of frame that the goal pose is with respect to.
        world_state: a viam.proto.common.WorldState
    """
    global MOCK_ROBOT_POSE
    MOCK_ROBOT_POSE = goal_pose  # mock -- as if the move succeeded
    return True


def viam_signal_find(viam_robot):
    """Do something with the robot to signal the find action"""
    return True
