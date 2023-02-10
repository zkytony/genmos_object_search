from genmos_object_search.oopomdp.domain.state import RobotState, ObjectState

def test_state_creation():
    # robot pose is represented as x, y, z, qx, qy, qz, qw
    r = RobotState("robot", (0, 0, 1, 2, 0, 0, 0, 1), (), None)
    assert r.loc == (0, 0, 1)
    assert r.is_2d is False
    assert r.pose == (0, 0, 1, 2, 0, 0, 0, 1)

    obj = ObjectState("car0", "car", (2.4, 23, 4.2))
    assert obj.loc == (2.4, 23, 4.2)
    assert obj.copy() == obj
