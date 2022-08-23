import pytest
import numpy as np
import pomdp_py
from sloop_object_search.oopomdp.domain.state import RobotState, ObjectState
from sloop_object_search.oopomdp.domain import action
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic3D
from sloop_object_search.utils.math import (euler_to_quat,
                                            approx_equal,
                                            R_to_quat,
                                            R_quat)

def _make_state(robot_state, state):
    """returns an OOState where the robot state is given
    by 'robot_state' while the other object states are
    given by 'state'."""
    object_states = state.object_states
    object_states["robot"] = robot_state
    next_state = pomdp_py.OOState(object_states)
    return next_state


@pytest.fixture
def positions():
    positions = set((x,y,z)
                    for x in range(-10,10)
                    for y in range(-10,10)
                    for z in range(-10,10))
    return positions

@pytest.fixture
def init_robot_state():
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    robot_state = RobotState("robot",
                             init_robot_pose,
                             (), None)
    return robot_state

@pytest.fixture
def target_states():
    return {"green_car": ObjectState("green_car", "car", (2, 2, 0)),
            "red_car": ObjectState("red_car", "car", (1, 5, 3))}

def test_quaternion_fact():
    """Test the fact that
    1. The rotation matrix for rotating first by z then by y
       is different from first rotating by y then by z.
       (order matters for 'roll-pitch-yaw' representation of rotation)
    2. quaternion represents a unique rotation matrix; As a result,
       it tells the difference between the two.
    """
    R1 = R_quat(*euler_to_quat(0, 0, 90))
    R2 = R_quat(*euler_to_quat(0, 90, 0))

    # this should be read as first rotate around y by 90, then around z by 90
    # because of the 'roll-pitch-yaw' convention, default for 'euler_to_quat'
    R3 = R_quat(*euler_to_quat(0, 90, 90))

    # R1*R2 --> first apply R2 then apply R1 --> equivalent to R3
    assert np.all(R_to_quat(R1*R2) == R_to_quat(R3))
    # R2*R1 --> first apply R1 then apply R2 --> DIFFERENT from R3
    assert not np.all(R_to_quat(R2*R1) == R_to_quat(R3))
    assert not np.all(R_to_quat(R2*R1) == R_to_quat(R1*R2)) # obvious

def test_robot_trans3d_without_detection_models(
        positions, init_robot_state, target_states):

    trobot = RobotTransBasic3D("robot",
                               lambda x: x in positions, {})
    state = pomdp_py.OOState({"robot": init_robot_state,
                              **target_states})
    action_axis = {a.motion_name: a
                   for a in action.basic_discrete_moves3d(scheme="axis")}
    action_forward = {a.motion_name: a
                   for a in action.basic_discrete_moves3d(scheme="forward")}

    # The robot moves along -z for 1 step
    next_robot_state = trobot.sample(state, action_axis["axis(-z)"])
    assert next_robot_state.loc == (2, -2, 3)

    # The robot's direction is -z. Moving forward will decrease it
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (2, -2, 2)
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (2, -2, 1)
    saved_next_robot_state = next_robot_state

    ### below is a sequence of tests where the robot first rotates
    ### and moves forward. This is used to test whether the relative rotation
    ### transition works correctly. In the beginning, the robot camera
    ### points to -z

    # Now, we rotate the robot camera around the z axis by 90 degrees -->
    # The robot's camera looks in the same direction, although its rotation matrix
    # is different.
    # **This is the reason why I don't like relative rotation actions.**
    # but for this test, it's fine.

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thz)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (2, -2, 0)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thy)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -2, 0)

    ### One more round of tests (a long sequence of rotations)
    next_robot_state = saved_next_robot_state
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thy)"])
    assert next_robot_state.loc == (2, -2, 1)
    # assert approx_equal(next_robot_state.pose[3:], tuple(euler_to_quat(0, 0, 90)), epsilon=1e-3)
    # going forward, the robot will still go in the -x direction
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -2, 1)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thz)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -3, 1)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thx)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -3, 0)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thx)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -2, 0)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thy)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -1, 0)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(-thx)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -1, -1)
    ### Now the camera is pointing to -z again.

    ### One more round of tests
    # Now, we rotate the robot camera around the z axis by 90 degrees -->
    # The robot's camera looks in the same direction, although its rotation matrix
    # is different.
    # **This is the reason why I don't like relative rotation actions.**
    # but for this test, it's fine.
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thz)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (1, -1, -2)

    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_axis["axis(+thy)"])
    state = _make_state(next_robot_state, state)
    next_robot_state = trobot.sample(state, action_forward["forward(forward)"])
    assert next_robot_state.loc == (0, -1, -2)
