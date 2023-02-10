import numpy as np
from genmos_object_search.utils.math import euler_to_quat, quat_to_euler, approx_equal


euler_angle = np.array([ -45.00859394,   89.99189756, -134.99140606])
quat_angle = np.array([ 0.4999, 0.5001, -0.5, 0.5001])
assert approx_equal(euler_to_quat(*euler_angle), quat_angle, epsilon=1e-3)
assert approx_equal(quat_to_euler(*quat_angle), euler_angle, epsilon=1e-3)
for i in range(100):
    euler_angle[0] += 5.0
    quat_angle = euler_to_quat(*euler_angle)
    euler_from_quat_angle = quat_to_euler(*quat_angle)
    quat_from_euler_from_quat_angle = euler_to_quat(*euler_from_quat_angle)
    if not approx_equal(euler_angle, euler_from_quat_angle, epsilon=1e-3):
        print("euler->quat->euler results in different euler")
        print("   {}".format(euler_angle))
        print("   {}".format(euler_from_quat_angle))
    if not approx_equal(quat_angle, quat_from_euler_from_quat_angle, epsilon=1e-3):
        print("quat->euler->quat results in different quat")
        print("   {}".format(quat_angle))
        print("   {}".format(quat_from_euler_from_quat_angle))

    print(euler_angle)
