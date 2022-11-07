import asyncio
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import numpy as np
import open3d as o3d
from viam_utils import (connect_viamlab_ur5,
                        viam_get_ee_pose,
                        viam_get_point_cloud_array,
                        viam_get_object_detections2d,
                        viam_detections2d_to_proto,
                        viam_get_image,
                        viam_move)
import viam.proto.common as v_pb2

from sloop_object_search.grpc.utils import proto_utils


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

async def test_viam_move_viamlab_ur5(viam_robot):
    # Built for the UR5 robot setup at Viam Lab
    table = v_pb2.Geometry(center=v_pb2.Pose(x=0, y=0, z=-20),
                           box=v_pb2.RectangularPrism(dims_mm=v_pb2.Vector3(x=2000, y=2000, z=40)))
    tableFrame = v_pb2.GeometriesInFrame(reference_frame="arm", geometries=[table])
    xARM = v_pb2.Geometry(center=v_pb2.Pose(x=600, y=0, z=0),
                          box=v_pb2.RectangularPrism(dims_mm=v_pb2.Vector3(x=200, y=200, z=600)))
    xARMFrame = v_pb2.GeometriesInFrame(reference_frame="arm_origin", geometries=[xARM])
    worldstate = v_pb2.WorldState(obstacles=[tableFrame, xARMFrame])

    pose = await viam_get_ee_pose(viam_robot)
    x,y,z,qx,qy,qz,qw = pose
    goal_pose = (x+0.1, y, z, qx, qy, qz, qw)
    print("moving robot arm.....")
    success = await viam_move(viam_robot, "arm", goal_pose, "arm_origin", worldstate)
    print("Successful?", success)
    print("-------------------------------------")


async def testall_viamlab_ur5():
    # Note on 11/06/2022 09:31AM: These tests are conducted at Viam Inc. using
    # their UR5 robot.
    print("Connecting to robot...")
    ur5robot = await connect_viamlab_ur5()
    print("Connected!")
    await test_viam_get_ee_pose(ur5robot)
    # await test_viam_get_point_cloud_array_to_proto(ur5robot)
    # await test_viam_get_image(ur5robot)
    # await test_viam_get_object_detections2d(ur5robot)
    await test_viam_move_viamlab_ur5(ur5robot)

    await ur5robot.close()

if __name__ == "__main__":
    asyncio.run(testall_viamlab_ur5())
