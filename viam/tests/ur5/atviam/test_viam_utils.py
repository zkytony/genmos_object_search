import asyncio
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../../../'))

import numpy as np
import open3d as o3d
from viam_utils import (viam_connect,
                        viam_get_point_cloud_array)

from sloop_object_search.grpc.utils import proto_utils


async def test_viam_get_point_cloud_array(viam_robot):
    cloud = await viam_get_point_cloud_array(viam_robot, debug=True)
    print("You should see Open3D window with reasonable point cloud. IF SO, pass.")

async def test_viam_get_point_cloud_array_to_proto(viam_robot):
    viam_cloud_arr = await viam_get_point_cloud_array(viam_robot, debug=False)

    # 'camera' is a made-up camera frame.
    cloud_pb = proto_utils.pointcloudproto_from_array(viam_cloud_arr, "camera")
    cloud_pb_arr = proto_utils.pointcloudproto_to_array(cloud_pb)

    viam_pcd = o3d.geometry.PointCloud()
    viam_pcd.points = o3d.utility.Vector3dVector(viam_cloud_arr)
    viam_pcd.points = o3d.utility.Vector3dVector(np.full((len(viam_cloud_arr), 3), (0.2, 0.2, 0.2)))

    my_pcd = o3d.geometry.PointCloud()
    my_pcd.points = o3d.utility.Vector3dVector(cloud_pb_arr)
    o3d.visualization.draw_geometries([viam_pcd, my_pcd])



async def testall():
    ur5robot = await viam_connect()
    # await test_viam_get_point_cloud_array(ur5robot)
    await test_viam_get_point_cloud_array_to_proto(ur5robot)

if __name__ == "__main__":
    asyncio.run(testall())
