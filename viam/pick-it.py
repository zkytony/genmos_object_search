import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.services.vision import VisionServiceClient
from viam.services.vision import VisModelConfig, VisModelType, Detection
from viam.components.camera import Camera

async def connect():
    creds = Credentials(
        type='robot-location-secret',
        payload='onq7ch977yafwxm7tuqlvfwuqlzmgu730mvn1exrr6w67t1s')
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address('macbook.496koy7yd1.local.viam.cloud:8080', opts)

async def main():
    robot = await connect()
    cam1 = Camera.from_robot(robot, "gripper-pi:combined")
    # grab Viam's vision service to add a TF-lite model for detection
    vision = VisionServiceClient.from_robot(robot)
    params = {
        "model_path": "/Users/bijanh/pick-it-bot/data/effdet0.tflite",
        "label_path": "/Users/bijanh/pick-it-bot/opencv/data/labels.txt",
        "num_threads": 1,
    }
    findThingDetector = VisModelConfig(name="find_thing", type=VisModelType("tflite_detector"), parameters=params)
    await vision.add_detector(findThingDetector)
    params = {
        "detector_name": "find_thing",
        "confidence_threshold_pct": 0.8,
    }
    findPersonDetector = VisModelConfig(name="find_thing_segmenter", type=VisModelType("detector_segmenter"), parameters=params)

    print('Resources:')
    print(robot.resource_names)
    while(True):
        pcs = await vision.get_object_point_clouds("gripper-pi:combined", "find_person_segmenter", "")
        print("number of points clouds:", len(pcs))
        if len(pcs) > 0:
            print(pcs[0].geometries)

    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())
