# note on 12/13/2022 09:03
# Code written specifically for the test at Viam Lab on the UR5 robot
#
# To run this test:
# -----------------
# 1. run in one terminal, run 'python -m sloop_object_search.grpc.server'
# 2. run in one terminal, run 'python test_run_search_viamlab_ur5.py'
# 3. run in one terminal, run 'roslaunch view_viam_search.launch'

import asyncio
import yaml
import os
import sys
import time

# Allow importing stuff from parent folder
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import constants

# Viam related
from utils import viam_utils
import viam.proto.common as v_pb2

# the core functionality
from sloop_mos_viam import SloopMosViam
import sloop_object_search.utils.math as math_utils


LAST_CANDIDATE = None

CANDIDATE_VIEWPOINTS = [
    {"name": "p1",
     "ee_pose": [-0.425, -0.417, 0.109, *viam_utils.ovec_to_quat(-0.04, -1.00, -0.02, 92.87)],
     "joint_positions": [22.38, -53.26, 101.41, -44.99, 24.63, 0.00]},

    {"name": "p2",
     "ee_pose": [-0.817, -0.232, 0.062, *viam_utils.ovec_to_quat(0.0, -1.00, -0.02, 90.00)],
     "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},

    {"name": "p3",
     "ee_pose": [-0.838, -0.066, -0.09, *viam_utils.ovec_to_quat(-0.72, -0.66, -0.21, 105.37)],
     "joint_positions": [-9.99, -10.01, 29.53, 0.00, 39.23, 0.01]},

    {"name": "p4",
     "ee_pose": [-0.242, -0.459, -0.203, *viam_utils.ovec_to_quat(-0.08, -0.96, -0.25, 93.00)],
     "joint_positions": [39.22, -71.06, 109.33, -17.72, 45.95, -11.60]},

    {"name": "p5",
     "ee_pose": [-0.190, -0.415, -0.149, *viam_utils.ovec_to_quat(-0.05, -0.93, -0.37, 100.00)],
     "joint_positions": [39.22, -71.06, 119.32, -17.72, 46.65, -11.61]},

    {"name": "p6",
     "ee_pose": [-0.340, -0.417, -0.20, *viam_utils.ovec_to_quat(-0.13, -0.99, -0.03, 90.00)],
     "joint_positions": [25.76, -41.64, 109.46, -62.94, 18.15, -4.65]},

    {"name": "p7",
     "ee_pose": [-0.816, -0.221, -0.44, *viam_utils.ovec_to_quat(-0.42, -0.89, -0.17, 87.00)],
     "joint_positions": [-0.01, 6.37, 1.01, 14.51, 26.95, -21.89]}
]


async def signal_find_func(viam_robot):
    # TODO: signal find
    success = await viam_utils.viam_signal_find(self.viam_robot)
    return success

async def move_viewpoint_ur5(dest, sloop_mos_viam):
    """moves the arm to goal pose.

    dest is the goal pose in my frame (not viam's).
    """
    global LAST_CANDIDATE
    viam_robot = sloop_mos_viam.viam_robot

    # account for frame difference (viam vs me)
    dest_viam = sloop_mos_viam.output_viam_pose(dest)

    candidates = CANDIDATE_VIEWPOINTS
    if LAST_CANDIDATE is not None:
        candidates = [v for v in CANDIDATE_VIEWPOINTS
                      if v["name"] != LAST_CANDIDATE]

    approx_dest = min(candidates,
                      key=lambda v: math_utils.euclidean_dist(
                          v["ee_pose"][:3], dest_viam[:3]))
    print(f"moving to joint positions in {approx_dest['name']}")
    await viam_utils.viam_move_to_joint_positions(
        viam_robot,
        approx_dest["joint_positions"],
        sloop_mos_viam.viam_names["arm"])
    LAST_CANDIDATE = approx_dest['name']
    # Whatever the movement is, we will try to rotate the last joint
    # to level it. Basically, get the rotation around the z axis (cuz
    # we are getting viam pose), and undo it by setting a joint angle
    # for the last joint.
    await viam_utils.viam_level_ur5gripper(viam_robot,
                                           sloop_mos_viam.viam_names["arm"])
    # sleep for 0.25 seconds for the detection result to come through
    time.sleep(0.25)
    return True


async def test_ur5e_viamlab():
    with open("../config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)

    print(">>>>>>><<<<<<<<>>>> viam connecting >><<<<<<<<>>>>>>>")
    ur5robot = await viam_utils.connect_viamlab_ur5()
    viam_info = {
        "names": {
            "color_camera": constants.COLOR_CAM,
            "detector": constants.DETECTOR,
            "arm": constants.ARM,
        },
        "move_viewpoint_func": move_viewpoint_ur5,
        "signal_find_func": signal_find_func
    }
    world_frame = "arm_origin"

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    sloop_viam = SloopMosViam()
    sloop_viam.setup(ur5robot, viam_info, config, world_frame)
    await sloop_viam.run()
    # await sloop_viam.stream_state()

if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
