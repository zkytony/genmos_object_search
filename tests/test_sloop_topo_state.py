from sloop.topo.domain.state import *

def test_state_init():
    s1 = TopoObjectState(1, "car", 4)
    srobot = TopoRobotState(3, 4)
    s = TopoJointState({1:s1, 3:srobot})
    assert s.s(1) == s1
    assert s.s(3) == srobot
    assert s1["id"] == 1
    assert srobot["id"] == 3
    assert srobot["status"].found_objects == tuple()
