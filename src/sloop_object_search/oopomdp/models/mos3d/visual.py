# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Plotting belief
import math
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np
from mos3d.util_viz import plot_voxels, CMAPS
from mos3d.util import remap
from pylab import rcParams
rcParams['figure.figsize'] = 4,4

from mos3d.planning.belief.octree import LOG

def _compute_alpha(p, vmin, vmax):
    if vmax - vmin > 0.0:
        return remap(p, vmin, vmax, 0, 1.0)
    else:
        return 1.0

def change_res(point, r1, r2):
    x,y,z = point
    return (x // (r2 // r1), y // (r2 // r1), z // (r2 // r1))

def plot_octree_belief(ax, octree_belief, robot_pose=None, cmap="jet", alpha=0.5,
                       edgecolor=None, linewidth=1):
    """If robot_pose is not None, then also plot robot pose"""
    # visualize the octree belief
    voxels = octree_belief.octree.collect_plotting_voxels()
    vp = [v[:3] for v in voxels]
    vr = [v[3] for v in voxels]
    if LOG:
        probs = [math.exp(octree_belief._probability(*change_res(v[:3], 1, v[3]),
                                                     v[3]))
                 for v in voxels]
    else:
        probs = [octree_belief._probability(*change_res(v[:3], 1, v[3]),
                                            v[3])
                 for v in voxels]
    vmin = min(probs)
    vmax = max(probs)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # probabilities to colors
    if alpha == "clarity":
        vc = [mcl.to_hex(m.to_rgba(p, alpha=_compute_alpha(p, vmin, vmax)),
                         keep_alpha=True)
              for p in probs]
    else:
        vc = [mcl.to_hex(m.to_rgba(p, alpha=alpha),
                         keep_alpha=True)
          for p in probs]

    # plot robot pose
    if robot_pose is not None:
        vp.append(robot_pose[:3])
        vr.append(1)  # robot pose is on the ground.
        vc.append([0,0,0,1])

    pc = plot_voxels(vp, vr, vc, ax=ax, edgecolor=edgecolor, linewidth=linewidth)

    # set array for color map so that it can be used as colorbar
    m.set_array(np.arange(vmin, vmax, 0.05))
    return m  # returns the color map
