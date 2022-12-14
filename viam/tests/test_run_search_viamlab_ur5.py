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


LAST_MOVE_IS_HOME = False

async def signal_find_func(viam_robot):
    # TODO: signal find
    success = await viam_utils.viam_signal_find(self.viam_robot)
    return success

async def move_viewpoint_ur5(dest, sloop_mos_viam):
    """moves the arm to goal pose.

    dest is the goal pose in my frame (not viam's).
    """
    global LAST_MOVE_IS_HOME

    viam_robot = sloop_mos_viam.viam_robot

    # account for frame difference (viam vs me)
    dest_viam = sloop_mos_viam.output_viam_pose(dest)

    # WARN: instead of doing going to the destination, go to
    # a closest pose known to work.
    # approx_dest_viam = min(WORKING_MOTION_POSES, key=lambda p: math_utils.euclidean_dist(p[:3], dest_viam[:3]))
    success = await viam_utils.viam_move(viam_robot,
                                         sloop_mos_viam.viam_names["arm"],
                                         dest_viam,
                                         sloop_mos_viam.world_frame,
                                         sloop_mos_viam.viam_world_state)
    if not success:
        print("viewpoint movement failed.")

        if not LAST_MOVE_IS_HOME:
            print(":::Move back to home configuration.")
            await viam_utils.viam_move_to_joint_positions(
                viam_robot,
                constants.UR5_HOME_CONFIG,
                sloop_mos_viam.viam_names["arm"])
            LAST_MOVE_IS_HOME = True

        else:
            # go to another filler pose just to not stay in place
            print(":::Move to alternative configuration.")
            await viam_utils.viam_move_to_joint_positions(
                viam_robot,
                constants.UR5_ALT_CONFIG,
                sloop_mos_viam.viam_names["arm"])
            LAST_MOVE_IS_HOME = False
        return success

    else:
        print("viewpoint movement succeeded.")
        LAST_MOVE_IS_HOME = False

    # Whatever the movement is, we will try to rotate the last joint
    # to level it. Basically, get the rotation around the z axis (cuz
    # we are getting viam pose), and undo it by setting a joint angle
    # for the last joint.
    viam_utils.viam_level_ur5gripper(viam_robot,
                                     sloop_mos_viam.viam_names["arm"])
    # sleep for 0.5 seconds for the detection result to come through
    time.sleep(0.5)
    return success


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
        "move_viewpoint_func": move_viewpoint_ur5
    }
    world_frame = "arm_origin"

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    sloop_viam = SloopMosViam()
    sloop_viam.setup(ur5robot, viam_info, config, world_frame)
    await sloop_viam.run()
    # await sloop_viam.stream_state()

if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
