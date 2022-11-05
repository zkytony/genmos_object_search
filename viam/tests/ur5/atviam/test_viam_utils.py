import asyncio
import os
import sys

sys.path.insert(0, os.path.join(ABS_PATH, '../../../'))

from viam_utils import (viam_connect,
                        viam_get_point_cloud_array)

async def test_viam_get_point_cloud_array(viam_robot):
    cloud = viam_get_point_cloud_array(viam_robot, debug=True)
    print("You should see Open3D window with reasonable point cloud. IF SO, pass.")


async def testall():
    ur5robot = viam_connect()
    test_viam_get_point_cloud_array(viam_robot)

if __name__ == "__main__":
    asyncio.run(testall())
