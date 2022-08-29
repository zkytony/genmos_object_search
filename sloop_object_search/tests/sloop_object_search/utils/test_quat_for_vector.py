import time
import vedo
import random
import numpy as np
from sloop_object_search.utils import math as math_utils


tt = 0
for i in range(5):
    x1 = random.uniform(-10, 10)
    y1 = random.uniform(-10, 10)
    z1 = random.uniform(-10, 10)

    x2 = random.uniform(-10, 10)
    y2 = random.uniform(-10, 10)
    z2 = random.uniform(-10, 10)

    v = math_utils.vec((x1,y1,z1), (x2,y2,z2))

    z = np.array([0, 0, 1])

    _s = time.time()
    q = math_utils.quat_between(z, v)
    tt += time.time() - _s

    # Draw the two points.  Then draw an arrow -> based on the euler angles from
    # the quaternion
    zp = np.dot(math_utils.R_quat(*q).as_matrix(), z)
    p1 = vedo.Point(pos=(x1,y1,z1), r=20, c='red')
    p2 = vedo.Point(pos=(x2,y2,z2), r=20, c='green')
    arrow1 = vedo.Arrow(startPoint=(0,0,0), endPoint=zp)
    arrow2 = vedo.Arrow(startPoint=(x1,y1,z1), endPoint=(x2,y2,z2), s=0.01, alpha=0.5)
    actors = [p1, p2, arrow1, arrow2]
    vedo.show(actors, axes=1).close()
print("time: {}s".format(tt))
