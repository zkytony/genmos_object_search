from genmos_object_search.oopomdp.domain.state import RobotState, ObjectState
from genmos_object_search.oopomdp.domain.observation import RobotObservation, ObjectDetection

def test_observation_creation():
    # robot pose is represented as x, y, z, qx, qy, qz, qw
    sr = RobotState("robot", (0, 0, 1, 2, 0, 0, 0, 1), (), None)
    zr = RobotObservation("robot", (0, 0, 1, 2, 0, 0, 0, 1), (), None)
    assert zr.loc == (0, 0, 1)
    assert zr.is_2d is False
    assert zr.pose == (0, 0, 1, 2, 0, 0, 0, 1)
    assert zr == RobotObservation.from_state(sr)

    sobj = ObjectState("car0", "car", (2.4, 23, 4.2))
    zobj = ObjectDetection("car0", (2.4, 23, 4.2))
    assert zobj.loc == sobj.loc
