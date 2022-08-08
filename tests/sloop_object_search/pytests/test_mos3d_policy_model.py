import pytest
import pomdp_py
from sloop_object_search.oopomdp.domain.state import RobotState, ObjectState
from sloop_object_search.oopomdp.domain.action import basic_discrete_moves3d, FindAction
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from sloop_object_search.oopomdp.models.policy_model import PolicyModelBasic3D
from sloop_object_search.utils.math import euler_to_quat

@pytest.fixture
def init_state():
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    robot_state = RobotState("robot",
                             init_robot_pose,
                             (), None)
    object_state = ObjectState("green_car", "car", (2, 0, 0))
    state = pomdp_py.OOState({"green_car": object_state,
                              robot_state.id: robot_state})
    return state

@pytest.fixture
def positions():
    positions = set((x,y,z)
                    for x in range(-10,10)
                    for y in range(-10,10)
                    for z in range(-10,10))
    return positions

def test_policy_model_basic_3d(init_state, positions):
    trobot = RobotTransBasic3D("robot", positions, {})
    movements = basic_discrete_moves3d()
    policy_model = PolicyModelBasic3D(["green_car"], trobot,
                                      movements)
    assert policy_model.get_all_actions(state=init_state)\
        == set(movements) | {FindAction()}
