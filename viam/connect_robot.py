import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

from viam.components.camera import Camera
from viam.services.vision import VisionServiceClient

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

    print('Resources:')
    print(robot.resource_names)

    camera = Camera.from_robot(robot, "comp-combined")
    image = await camera.get_image()
    image.save("foo.png")

    print("----------------------------------")

    vision = VisionServiceClient.from_robot(robot)
    segmenter_names = await vision.get_segmenter_names()
    print(segmenter_names)


    print("HEEELO")

    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())
