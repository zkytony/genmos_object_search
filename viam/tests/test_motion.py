import asyncio
from viam.proto.common import Pose, PoseInFrame, Geometry, RectangularPrism, GeometriesInFrame, Vector3, WorldState
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.services.motion import MotionServiceClient
from viam.components.arm import Arm
from pprint import pprint

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

    arm_name = "arm"

    #pprint(await robot.get_frame_system_config())
    # create a geometry representing the table to which the arm is attached
    table = Geometry(center=Pose(x=0, y=0, z=-20), box=RectangularPrism(dims_mm=Vector3(x=2000, y=2000, z=40)))
    tableFrame = GeometriesInFrame(reference_frame=arm_name, geometries=[table])

    # create a geometry 200mm behind the arm to block it from hitting the xarm
    xARM = Geometry(center=Pose(x=600, y=0, z=0), box=RectangularPrism(dims_mm=Vector3(x=200, y=200, z=600)))
    xARMFrame = GeometriesInFrame(reference_frame="arm_origin", geometries=[xARM])

    worldstate = WorldState(obstacles=[tableFrame, xARMFrame])

    arm = Arm.from_robot(robot, "arm")
    pose = await arm.get_end_position()
    pose.x -= 100
    motion = MotionServiceClient.from_robot(robot)
    for resname in robot.resource_names:
        if resname.name == "arm":
            print (resname)
            move = await motion.move(component_name=resname, destination =
                                     PoseInFrame(reference_frame="arm_origin", pose=pose),
                                     world_state=worldstate)

    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())
