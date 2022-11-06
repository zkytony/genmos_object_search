import asyncio
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))

import numpy as np
import open3d as o3d
from viam_utils import (connect_viamlab_ur5,
                        viam_get_ee_pose,
                        viam_get_point_cloud_array)

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
    print("----------------------")

async def testall_viamlab_ur5():
    # Note on 11/06/2022 09:31AM: These tests are conducted at Viam Inc. using
    # their UR5 robot.
    ur5robot = await connect_viamlab_ur5()
    await test_viam_get_ee_pose(ur5robot)
    await test_viam_get_point_cloud_array_to_proto(ur5robot)


if __name__ == "__main__":
    asyncio.run(testall_viamlab_ur5())
