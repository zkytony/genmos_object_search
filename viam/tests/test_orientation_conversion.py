import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../'))
from viam_utils import OrientationVector, Quaternion, Vector3
from sloop_object_search.utils import math as math_utils

pose = {
    "x": 462.41402300101993,
    "y": 340.22520636265983,
    "z": 625.99252608964252,
    "o_x": -0.16668243758578755,
    "o_y": 0.93190409908252192,
    "o_z": -0.322136174798259,
    "theta": 71.308462217232076
}

ovec = OrientationVector(Vector3(pose['o_x'], pose['o_y'], pose['o_z']), math_utils.to_rad(pose['theta']))
qq = Quaternion.from_orientation_vector(ovec)
ovec2 = qq.to_orientation_vector()
qq2 = Quaternion.from_orientation_vector(ovec2)
print(qq)
print(qq2)
