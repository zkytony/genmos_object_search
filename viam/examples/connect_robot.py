import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

from viam.components.camera import Camera
from viam.services.vision import VisionServiceClient, VisModelConfig, VisModelType

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

    print("----------------------------------")

    # grab Viam's vision service to add a TF-lite model for detection
    vision = VisionServiceClient.from_robot(robot)
    params = {
        "model_path": "/home/kaiyu/repo/robotdev/shared/ros/sloop_object_search/viam/models/effdet0.tflite",
        "label_path": "/home/kaiyu/repo/robotdev/shared/ros/sloop_object_search/viam/models/effdet0_labels.txt",
        "num_threads": 1,
    }
    findThingDetector = VisModelConfig(
        name="find_thing", type=VisModelType("tflite_detector"), parameters=params)
    await vision.add_detector(findThingDetector)

    print(await vision.get_detector_names())

    params = {
        "detector_name": "find_thing",
        "confidence_threshold_pct": 0.8,
    }
    findPersonDetector = VisModelConfig(name="find_thing_segmenter", type=VisModelType("detector_segmenter"), parameters=params)

    print('Resources:')
    print(robot.resource_names)
    while(True):
        pcs = await vision.get_object_point_clouds(
            "comp-combined", "find_thing_segmenter")
        print("number of points clouds:", len(pcs))
        if len(pcs) > 0:
            print(pcs[0].geometries)

    print("HEEELO")

    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())
