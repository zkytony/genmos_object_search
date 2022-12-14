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

# Allow importing stuff from parent folder
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import constants

# Viam related
from utils import viam_utils
import viam.proto.common as v_pb2

# the core functionality
from sloop_mos_viam import SloopMosViam


async def test_ur5e_viamlab():
    with open("../config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)

    print(">>>>>>><<<<<<<<>>>> viam connecting >><<<<<<<<>>>>>>>")
    ur5robot = await viam_utils.connect_viamlab_ur5()
    viam_names = {
        "color_camera": constants.COLOR_CAM,
        "detector": constants.DETECTOR,
        "arm": constants.ARM
    }
    world_frame = "arm_origin"

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    sloop_viam = SloopMosViam()
    sloop_viam.setup(ur5robot, viam_names, config, world_frame)
    await sloop_viam.run()

if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
