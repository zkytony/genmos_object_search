import pytest
import pomdp_py

from genmos_object_search.oopomdp.domain.state import RobotState, ObjectState
from genmos_object_search.oopomdp.domain.observation import Voxel
from genmos_object_search.oopomdp.domain.action import FindAction
from genmos_object_search.oopomdp.models.detection_models import FrustumVoxelAlphaBeta
from genmos_object_search.oopomdp.models.reward_model import GoalBasedRewardModel
from genmos_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from genmos_object_search.utils.math import euler_to_quat

@pytest.fixture
def init_robot_state():
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    robot_state = RobotState("robot",
                             init_robot_pose,
                             (), None)
    return robot_state

@pytest.fixture
def positions():
    positions = set((x,y,z)
                    for x in range(-10,10)
                    for y in range(-10,10)
                    for z in range(-10,10))
    return positions


def test_reward_model(init_robot_state, positions):
    objid = "green_car"
    object_state_visible = ObjectState(objid, "car", (2, 0, 0))
    object_state_not_visible = ObjectState(objid, "car", (2, 0, -2))

    # Reliable camera -------------------------------------------
    ALPHA=1e5
    BETA=0.1
    frustum_params = dict(fov=60, aspect_ratio=1.0, near=1, far=5,
                          default_look=(0,0,-1))
    quality_params = (ALPHA, BETA)
    detection_model = FrustumVoxelAlphaBeta(objid, frustum_params, quality_params)

    # transition model
    trobot = RobotTransBasic3D("robot", positions,
                               {objid: detection_model})

    # Test 'find' when target is visible, and within angular tolerance (30 degrees, default)
    state = pomdp_py.OOState({objid: object_state_visible,
                              init_robot_state.id: init_robot_state})
    action = FindAction()
    next_robot_state = trobot.sample(state, action)
    assert next_robot_state.objects_found == (objid,)

    # Test 'find' when target is not visible
    state = pomdp_py.OOState({objid: object_state_not_visible,
                              init_robot_state.id: init_robot_state})
    action = FindAction()
    next_robot_state = trobot.sample(state, action)
    assert len(next_robot_state.objects_found) == 0
