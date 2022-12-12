import asyncio
import pytest
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import numpy as np
import open3d as o3d
from utils.viam_utils import (connect_viamlab_ur5,
                              viam_get_ee_pose,
                              viam_get_point_cloud_array,
                              viam_get_object_detections2d,
                              viam_get_joint_positions,
                              viam_move_to_joint_positions,
                              viam_detections2d_to_proto,
                              viam_get_image,
                              viam_move,
                              ovec_to_quat,
                              quat_to_ovec)
from utils import viam_utils
from constants import UR5_HOME_CONFIG

from viam.components.arm import Arm

import viam.proto.common as v_pb2
from viam.proto.common import (Pose, PoseInFrame, Geometry, RectangularPrism,
                               GeometriesInFrame, Vector3, WorldState)

from sloop_object_search.grpc.utils import proto_utils


def test_quat_ovec_conversion():
    pose = {
        "x": 462.41402300101993,
        "y": 340.22520636265983,
        "z": 625.99252608964252,
        "o_x": -0.16668243758578755,
        "o_y": 0.93190409908252192,
        "o_z": -0.322136174798259,
        "theta": 71.308462217232076
    }
    ovec = (pose['o_x'], pose['o_y'], pose['o_z'], pose['theta'])
    quat = ovec_to_quat(*ovec)
    ovec2 = quat_to_ovec(*quat)
    quat2 = ovec_to_quat(*ovec2)
    assert ovec2 == pytest.approx(ovec)
    assert quat2 == pytest.approx(quat)

async def test_viam_get_point_cloud_array_to_proto(viam_robot):
    # THIS TEST MAY FAIL BECAUSE:
    #  -- The depth cloud is not accessible (you will get "Unknown desc = image: unknown format")
    viam_cloud_arr = await viam_get_point_cloud_array(viam_robot)

    # 'camera' is a made-up camera frame.
    cloud_pb = proto_utils.pointcloudproto_from_array(viam_cloud_arr, "camera")
    cloud_pb_arr = proto_utils.pointcloudproto_to_array(cloud_pb)

    viam_pcd = o3d.geometry.PointCloud()
    viam_pcd.points = o3d.utility.Vector3dVector(viam_cloud_arr)
    viam_pcd.colors = o3d.utility.Vector3dVector(np.full((len(viam_cloud_arr), 3), (0.2, 0.2, 0.2)))

    my_pcd = o3d.geometry.PointCloud()
    my_pcd.points = o3d.utility.Vector3dVector(cloud_pb_arr)

    viz = o3d.visualization.Visualizer()
    viz.create_window()
    viz.add_geometry(viam_pcd)
    viz.add_geometry(my_pcd)
    opt = viz.get_render_option()
    opt.show_coordinate_frame = True
    viz.run()
    viz.destroy_window()
    print("----------------------")


async def test_viam_get_ee_pose(viam_robot):
    pose = await viam_get_ee_pose(viam_robot)
    print(pose)
    pose_pb = proto_utils.posetuple_to_poseproto(pose)
    print(pose_pb)
    print("----------------------")

async def test_viam_get_joint_positions(viam_robot):
    joints = await viam_get_joint_positions(viam_robot)
    print(joints)
    print("----------------------")


async def test_viam_get_image(viam_robot):
    image = await viam_get_image(viam_robot, "gripper:color-cam")
    image.save("foo.png")
    print("image saved")
    image = await viam_get_image(viam_robot, "segmenter-cam")
    image.save("foo-seg.png")
    print("segmenter image saved")
    print("----------------------")

async def test_viam_get_object_detections2d(viam_robot):
    detections = await viam_get_object_detections2d(viam_robot, confidence_thres=0.5)
    print(detections)
    print("::: convert to my ObjectDetectionArray proto :::")
    detections_pb = viam_detections2d_to_proto("robot0", detections)
    print(detections_pb)
    print("----------------------")

async def test_viam_move_to_joint_pos_viamlab_ur5(viam_robot, arm_name="arm"):
    await viam_move_to_joint_positions(viam_robot, UR5_HOME_CONFIG, arm_name)
    print("----------------------")


async def test_viam_move_viamlab_ur5(viam_robot, arm_name="arm", world_frame="world"):
    """testing motion planning service (end-effector) at Viam lab with the UR5 robot.
    Also tests move to joint positions through returning to home."""
    def build_world_state_viamlab_ur5():
        # World State... defined in test_motion_nick_viam.py
        # create a geometry 200mm behind the arm to block it from hitting the xarm
        wallBehind = Geometry(center=Pose(x=200, y=0, z=500), box=RectangularPrism(dims_mm=Vector3(x=80, y=2000, z=2000)))
        wallBehindFrame = GeometriesInFrame(reference_frame=world_frame, geometries=[wallBehind])

        # create a geometry representing the table I am sitting at
        mytable = Geometry(center=Pose(x=-450, y=0, z=-266), box=RectangularPrism(dims_mm=Vector3(x=900, y=2000, z=100)))
        tableFrame = GeometriesInFrame(reference_frame=world_frame, geometries=[mytable])

        # create a geometry representing the table to the arm is attached
        mount = Geometry(center=Pose(x=300, y=0, z=-500), box=RectangularPrism(dims_mm=Vector3(x=700, y=1000, z=1000)))
        mountFrame = GeometriesInFrame(reference_frame=world_frame, geometries=[mount])

        worldstate = WorldState(obstacles=[tableFrame, wallBehindFrame, mountFrame] )
        return worldstate

    # Adapting the examples from Nick in test_motion_nick_viam.py
    world_state = build_world_state_viamlab_ur5()

    test_goals = [
        (-0.6, -0.4, 0.06, *ovec_to_quat(0, -1, 0, 0)),
        (-0.6, -0.4, 1.60, *ovec_to_quat(0, -1, 0, 0)),
        (-0.6, -0.4, 0.60, *ovec_to_quat(0, -1, 0, 0)),
        (-0.7, -0.4, 0.60, *ovec_to_quat(0, -1, 0, 0)),
        (-0.5, -1.3, 0.53, *ovec_to_quat(0.05, -0.02,  -1.00,  70.20)),
        (-0.42, -1.3, 0.53, *ovec_to_quat(0.05, -0.02,  -1.00,  70.20))
    ]

    arm = Arm.from_robot(viam_robot, arm_name)
    for i, goal_pose in enumerate(test_goals):
        print(f"---------test {i}---------")
        pose = await arm.get_end_position()
        # note all rotations are printed in quaternion
        print("Start EE pose:", viam_utils.pose_ovec_to_quat(pose))
        success = await viam_move(viam_robot,
                                  arm_name,
                                  goal_pose,
                                  world_frame,
                                  world_state)
        print("Goal EE pose:", goal_pose)
        if not success:
            print('motion planning failed.')
        else:
            pose = await arm.get_end_position()
            print("Actual EE pose:", viam_utils.pose_ovec_to_quat(pose))
    print("-------------------------------------")


async def testall_viamlab_ur5():
    test_quat_ovec_conversion()

    print("Connecting to robot...")
    ur5robot = await connect_viamlab_ur5()
    print("Connected!")

    await test_viam_get_joint_positions(ur5robot)
    await test_viam_move_viamlab_ur5(ur5robot)
    await test_viam_move_to_joint_pos_viamlab_ur5(ur5robot)



    # # Note on 11/06/2022 09:31AM: These tests are conducted at Viam Inc. using
    # # their UR5 robot.
    # print("Connecting to robot...")
    # ur5robot = await connect_viamlab_ur5()
    # print("Connected!")
    # await test_viam_get_ee_pose(ur5robot)
    # # await test_viam_get_point_cloud_array_to_proto(ur5robot)
    # # await test_viam_get_image(ur5robot)
    # # await test_viam_get_object_detections2d(ur5robot)
    # await test_viam_move_viamlab_ur5(ur5robot)

    # await ur5robot.close()

if __name__ == "__main__":
    asyncio.run(testall_viamlab_ur5())
