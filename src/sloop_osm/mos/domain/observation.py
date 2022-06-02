import sloop

class ObjectDetection(sloop.domain.observation.ObjectObservation):
    """Observation of a target object's location"""
    NULL = None  # empty
    def __init__(self, objid, pose):
        super().__init__(objid, pose)

    @property
    def pose(self):
        return self.data

    def __str__(self):
        return f"{self.objid}({self.objid}, {self.pose})"

    @property
    def id(self):
        return self.objid


class RobotObservation(sloop.domain.observation.RobotObservation):
    def __init__(self, robot_id, robot_pose, objects_found, camera_direction):
        self.robot_id = robot_id
        self.pose = robot_pose
        self.objects_found = objects_found
        self.camera_direction = camera_direction
        super().__init__((self.robot_id, self.pose, self.objects_found, self.camera_direction))

    def __str__(self):
        return f"{self.robot_id}({self.pose, self.camera_direction, self.objects_found})"
