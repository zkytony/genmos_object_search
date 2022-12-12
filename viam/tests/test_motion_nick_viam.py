import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
import time

from viam.services.motion import MotionServiceClient
from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame, Geometry, GeometriesInFrame, RectangularPrism, WorldState
from viam.gen.common.v1.common_pb2 import Vector3

async def connect():
    creds = Credentials(
        type='robot-location-secret',
        payload='gm1rjqe84nt8p64ln6r1jyf5hc3tdnc2jywojoykvk56d0qa')
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address('viam-test-bot-main.tcyat99x8y.viam.cloud', opts)

async def main():
    robot = await connect()

    # create a geometry 200mm behind the arm to block it from hitting the xarm
    wallBehind = Geometry(center=Pose(x=200, y=0, z=500), box=RectangularPrism(dims_mm=Vector3(x=80, y=2000, z=2000)))
    wallBehindFrame = GeometriesInFrame(reference_frame="world", geometries=[wallBehind])

    # create a geometry protecting myself from arm movement
    # leftWall = Geometry(center=Pose(x=0, y=350, z=0), box=RectangularPrism(dims_mm=Vector3(x=2000, y=100, z=2000)))
    # leftWallFrame = GeometriesInFrame(reference_frame="world", geometries=[leftWall])


    # create a geometry representing the table I am sitting at
    mytable = Geometry(center=Pose(x=-450, y=0, z=-266), box=RectangularPrism(dims_mm=Vector3(x=900, y=2000, z=100)))
    tableFrame = GeometriesInFrame(reference_frame="world", geometries=[mytable])

    # create a geometry representing the table to the arm is attached
    mount = Geometry(center=Pose(x=300, y=0, z=-500), box=RectangularPrism(dims_mm=Vector3(x=700, y=1000, z=1000)))
    mountFrame = GeometriesInFrame(reference_frame="world", geometries=[mount])

    worldstate = WorldState(obstacles=[tableFrame, wallBehindFrame, mountFrame] )

    # get resource name for ur5e


    # pose using motion service
    motion = MotionServiceClient.from_robot(robot)
    # print(resourceName)
    resourceName = getMe(robot.resource_names)
    getMe(robot.resource_names)
    myPose = await motion.get_pose(resourceName, "")
    # myPose.pose.x += 100

    #last-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-up
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=160, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-down
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=-60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-left
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-700, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-right
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-500, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #last-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-600, y=-400, z=60, o_y=-1.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    # down
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-623.13, y=-134.92, z=535.05, o_z=-1.00))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #up
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-423.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #left
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-234.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #right
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-34.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-523.13, y=-134.92, z=535.05, o_z=-1.00, o_x=0.05,  o_y=-0.02, theta=70.20))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-482, y=550, z=400, o_z=0.00, o_x=0.0,  o_y=1.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-up
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-482, y=550, z=500, o_z=0.00, o_x=0.0,  o_y=1.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-482, y=550, z=400, o_z=0.00, o_x=0.0,  o_y=1.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-down
    print("1")
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=300, o_y=1.0, o_x=0.0, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=400, o_y=1.0, o_x=0.0, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-up
    print("2")
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=500, o_y=1.0, o_x=0.01, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=400, o_y=1.0, o_x=0.01, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)


    #new-right
    print("3")
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-650, y=500, z=400, o_y=1.0, o_x=0.0, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=400, o_z=0.00, o_x=0.0,  o_y=1.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    time.sleep(1)

    #new-left
    print("4")
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-450, y=500, z=400, o_y=1.0, o_x=0.0, o_z=0.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)

    #new-og
    searchPose = PoseInFrame(reference_frame="world", pose=Pose(x=-550, y=500, z=400, o_z=0.00, o_x=0.0,  o_y=1.0, theta=0.0))
    await motion.move(component_name=resourceName, destination = searchPose, world_state=worldstate)


    # move back to home
    await motion.move(component_name=resourceName, destination = myPose)


    await robot.close()

def getMe(iterate):
    for resname in iterate:
        print(resname.name)
        if resname.name == "arm":
            resourceName = resname
            print(resourceName)
    return resourceName

if __name__ == '__main__':
    asyncio.run(main())
