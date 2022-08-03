from sloop_object_search.oopomdp.domain import action

def test_making_actions():
    actions_axis = action.basic_discrete_moves3d(step_size=2, scheme="axis")
    actions_forward = action.basic_discrete_moves3d(step_size=2, scheme="forward")

    actions_axis = {a.motion_name: a for a in actions_axis}
    actions_forward = {a.motion_name: a for a in actions_forward}
    assert actions_axis["axis(-x)"].motion[0] == (-2, 0, 0)
    assert actions_axis["axis(+thx)"].motion[1] == (90.0, 0, 0)
    assert actions_forward["forward(forward)"].motion[0] == 2
    assert actions_forward["forward(+thx)"].motion[1] == (90.0, 0, 0)
