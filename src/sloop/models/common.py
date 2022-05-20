
class SensorModel:
    IS_3D = False
    def in_range(self, point, sensor_pose):
        raise NotImplementedError

    def in_range_facing(self, point, sensor_pose,
                        angular_tolerance=15):
        """Returns True if the point is within the field of view,
        AND the sensor pose is facing the object directly,
        with some angular tolerance"""
        raise NotImplementedError
