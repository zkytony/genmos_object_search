import math
import yaml
import time
import numpy as np
import open3d as o3d
import asyncio
from dataclasses import dataclass

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionServiceClient, VisModelConfig, VisModelType
from viam.services.motion import MotionServiceClient

import viam.proto.common as v_pb2
import viam.proto.component.arm as varm_pb2


async def connect_viamlab_ur5():
    creds = Credentials(
        type='robot-location-secret',
        payload='gm1rjqe84nt8p64ln6r1jyf5hc3tdnc2jywojoykvk56d0qa')
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address('viam-test-bot-main.tcyat99x8y.viam.cloud', opts)


async def viam_get_ee_pose(viam_robot, arm_name="arm"):
    """return current end-effector pose in world
    frame through Viam.
    Return type: tuple (x,y,z,qx,qy,qz,qw)
    Note that viam's positions units are in milimeters.
    We will convert them into meters (more familiar with me)"""
    #NOTE!!! BELOW DOES NOT GIVE YOU THE TRUE EE
    #ON THE GRIPPER OF THE UR5 ROBOT AT VIAM LAB
    #BUT THE END OF THE ARM WITHOUT GRIPPER. THIS
    #IS BECAUSE THE GRIPPER IS A SEPARATE COMPUTER
    #AND CURRENTLY THERE IS A BUG IN VIAM TO GET
    #THAT FRAME. ALSO, SHOULD USE MOTIONSERVICE
    #INSTEAD OF ARM BUT CANT BECAUSE OF THAT BUG.
    arm = Arm.from_robot(viam_robot, arm_name)
    pose_w_ovec = await arm.get_end_position()

    return pose_w_ovec

async def viam_get_pose(viam_robot, component_name, frame):
    """
    Returns the pose of the component in the given frame.
    Args:x
        component_name (str): name of the component
        frame (str): name of the frame to express the component's pose
    Returns:
        tuple (x,y,z,qx,qy,qz,qw)
    Note that viam's positions units are in milimeters.
    We will convert them into meters (more familiar with me)
    """
    motion = MotionServiceClient.from_robot(viam_robot)
    comp_resname = viam_get_resname(viam_robot, component_name)
    pose_w_ovec = (await motion.get_pose(comp_resname, frame)).pose
    return pose_w_ovec

def viam_get_resname(viam_robot, component_name):
    """returns the ResourceName corresponding to component name"""
    comp_resname = None
    for resname in viam_robot.resource_names:
        if resname.name == component_name:
            comp_resname = resname
    return comp_resname


async def test():
    # Note on 12/12/2022 15:42: These tests are conducted at Viam Inc. using
    print("Connecting to robot...")
    ur5robot = await connect_viamlab_ur5()
    print("Connected!")

    # Testing perception
    print(await viam_get_ee_pose(ur5robot))
    print(await viam_get_pose(ur5robot, "arm", "world"))
    await ur5robot.close()

if __name__ == "__main__":
    asyncio.run(test())
