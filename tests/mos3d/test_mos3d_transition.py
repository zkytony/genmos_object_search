import pomdp_py
from sloop_object_search.oopomdp.domain.state import RobotState, ObjectState
from sloop_object_search.oopomdp.domain import action
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from sloop_object_search.utils.math import euler_to_quat

def test_robot_trans3d_without_detection_models():
    # Test without detection models
    positions = set((x,y,z)
                    for x in range(-10,10)
                    for y in range(-10,10)
                    for z in range(-10,10))
    trobot = RobotTransBasic3D("robot", positions, {})

    # init robot pose
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    robot_state = RobotState("robot",
                             init_robot_pose,
                             (), None)
    state = pomdp_py.OOState({"robot": robot_state,
                              "green_car": ObjectState("green_car", "car", (2, 2, 0)),
                              "red_car": ObjectState("red_car", "car", (1, 5, 3))})

    action_axis = {a.motion_name: a
                   for a in action.basic_discrete_moves3d(scheme="axis")}

    action_forward = {a.motion_name: a
                   for a in action.basic_discrete_moves3d(scheme="forward")}

    next_robot_state = trobot.sample(state, action_axis["axis(-z)"])
    assert next_robot_state.loc == (2, -2, 3)
