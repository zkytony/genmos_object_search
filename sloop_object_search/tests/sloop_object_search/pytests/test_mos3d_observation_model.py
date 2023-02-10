import pytest
from genmos_object_search.oopomdp.models.detection_models import FrustumVoxelAlphaBeta
from genmos_object_search.oopomdp.domain.state import RobotState, ObjectState
from genmos_object_search.oopomdp.domain.observation import Voxel
from genmos_object_search.utils.math import euler_to_quat

@pytest.fixture
def init_robot_state():
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    robot_state = RobotState("robot",
                             init_robot_pose,
                             (), None)
    return robot_state

def test_frustum_alpha_beta_model(init_robot_state):
    objid = "green_car"
    object_state_visible = ObjectState(objid, "car", (2, 0, 0))
    object_state_not_visible = ObjectState(objid, "car", (2, 0, -2))

    # Reliable camera -------------------------------------------
    ALPHA=1e5
    BETA=0.1
    frustum_params = dict(fov=60, aspect_ratio=1.0, near=1, far=5,
                          default_look=(0,0,-1))
    quality_params = (ALPHA, BETA)
    model = FrustumVoxelAlphaBeta(objid, frustum_params, quality_params)

    obz = model.sample(object_state_visible, init_robot_state)
    assert obz.label == object_state_visible.id

    obz = model.sample(object_state_not_visible, init_robot_state)
    assert obz.label == Voxel.UNKNOWN

    # Unrealiable camera ----------------------------------------
    ALPHA=0.1
    BETA=1000
    objid = "green_car"
    frustum_params = dict(fov=60, aspect_ratio=1.0, near=1, far=5,
                          default_look=(0,0,-1))
    quality_params = (ALPHA, BETA)
    model = FrustumVoxelAlphaBeta(objid, frustum_params, quality_params)

    obz = model.sample(object_state_visible, init_robot_state)
    assert obz.label == Voxel.OTHER

    obz = model.sample(object_state_not_visible, init_robot_state)
    assert obz.label == Voxel.UNKNOWN
