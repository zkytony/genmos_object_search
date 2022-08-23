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


import time
import math
import pytest
import matplotlib.pyplot as plt

from pomdp_py import OOTransitionModel

from sloop_object_search.oopomdp.models.octree_belief\
    import (OctreeBelief, OctreeDistribution, update_octree_belief,
            Octree, OctNode, DEFAULT_VAL,
            plot_octree_belief)

from sloop_object_search.oopomdp import ObjectState
from sloop_object_search.utils.math import approx_equal
from sloop_object_search.oopomdp.models.sensors import FrustumCamera
from sloop_object_search.oopomdp.domain.observation import FovVoxels, Voxel


TEST_ALPHA = 100000
TEST_BETA = 0.00001


@pytest.fixture
def octree_belief():
    octree_dist = OctreeDistribution((16, 16, 16))
    octree_belief = OctreeBelief(1, "cube", octree_dist)
    return octree_belief

def test_basics(octree_belief):
    # Test probability; __getitem__
    print("** Testing Basics")
    assert abs(octree_belief.octree_dist.prob_at(0,0,1,1) - 1./(16**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,2) - 1./(8**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,4) - 1./(4**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,8) - 1./(2**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,16) - 1./(1**3)) <= 1e-6
    print(octree_belief.octree_dist._known_voxels)

    # assign a certain (unnormalized) probability to a node. Note that
    # in the assertion statements, the expected probability can be
    # computed based on the probability at the ground resolution level,
    # which is how octree belief is defined.
    octree_belief[ObjectState(1, "cube", (0,0,1), res=1)] = TEST_ALPHA
    print(octree_belief.octree_dist._known_voxels)
    assert abs(octree_belief.octree_dist.prob_at(0,0,1,1) - TEST_ALPHA/(TEST_ALPHA + 16**3 - 1)) <= 1e-6

    # 0,0,1 at res=1 is a child in 0,0,0 at res=2
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,2) - (TEST_ALPHA+7)/(TEST_ALPHA + 7 + 16**3 - 8)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,1,2) - 8/(TEST_ALPHA + 7 + 16**3 - 8)) <= 1e-6

    # this should always be true
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,16) - 1./(1**3)) <= 1e-6

    # set it back
    octree_belief[ObjectState(1, "cube", (0,0,1), res=1)] = DEFAULT_VAL
    print(octree_belief.octree_dist._known_voxels)
    assert abs(octree_belief.octree_dist.prob_at(0,0,1,1) - 1./(16**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,2) - 1./(8**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,4) - 1./(4**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,8) - 1./(2**3)) <= 1e-6
    assert abs(octree_belief.octree_dist.prob_at(0,0,0,16) - 1./(1**3)) <= 1e-6

def test_assign_prior1(octree_belief):
    print("** Testing Prior assignment (1)")
    print("[start]")
    state = ObjectState(1, "cube", (1,1,1), res=2)
    print("Probability at")
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    octree_belief.assign(state, TEST_ALPHA/10)
    print("MPE: %s" % octree_belief.mpe())
    assert octree_belief.octree_dist.prob_at(0,0,0,1) < octree_belief.octree_dist.prob_at(3,3,3,1)
    assert octree_belief.mpe(res=2).loc == (1,1,1)
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief.octree_dist.prob_at(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_assign_prior2(octree_belief):
    print("** Testing Prior assignment (2)")
    print("[start]")
    state = ObjectState(1, "cube", (5,6,7), res=2)
    print("Probability at")
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    octree_belief.assign(state, TEST_ALPHA/10)
    print("MPE: %s" % octree_belief.mpe())
    assert octree_belief.octree_dist.prob_at(0,0,0,1) < octree_belief.octree_dist.prob_at(11,13,15,1)
    assert octree_belief.mpe(res=2).loc == (5,6,7)
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief.octree_dist.prob_at(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_assign_prior3(octree_belief):
    print("** Testing Prior assignment (3) ZERO Prior")
    print("[start]")
    state = ObjectState(1, "cube", (5,6,7), res=2)
    print("Probability at")
    print("(5, 6, 7, 2): %.5f" % octree_belief.octree_dist.prob_at(5, 6, 7, 2))
    print("(11,13,15,1): %.5f" % octree_belief.octree_dist.prob_at(11,13,15,1))
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("MPE: %s" % octree_belief.mpe())
    print("assigning high probability to (1,1,1,2)...")
    octree_belief.assign(state, 0)
    print("MPE: %s" % octree_belief.mpe())
    assert octree_belief.octree_dist.prob_at(0,0,0,1) > octree_belief.octree_dist.prob_at(11,13,15,1)
    assert octree_belief.mpe(res=2).loc != (5,6,7)
    print("(5, 6, 7, 2): %.5f" % octree_belief.octree_dist.prob_at(5, 6, 7, 2))
    print("(11,13,15,1): %.5f" % octree_belief.octree_dist.prob_at(11,13,15,1))
    print("(1,1,1,2): %.5f" % octree_belief.octree_dist.prob_at(1,1,1,2))
    print("(0,0,0,1): %.5f" % octree_belief.octree_dist.prob_at(0,0,0,1))
    print("(3,3,3,1): %.5f" % octree_belief.octree_dist.prob_at(3,3,3,1))
    print("**** Sub test:")
    test_mpe_random(octree_belief, res=2)  # MPE/random at resolution 2
    print("[end]")

def test_mpe_random(octree_belief, res=1):
    def test_round(octree_belief):
        print("-- Round --")
        mpe_state = octree_belief.mpe(res=res)
        mpe_prob = octree_belief[mpe_state]
        print(mpe_state)

        results = []
        num_same = 0
        for i in range(10000):
            rnd_state = octree_belief.random(res=res)
            results.append(rnd_state)
            if mpe_state == rnd_state:
                num_same += 1
        print("Expected probability: %.5f; Actual frequency: %.5f"
              % (mpe_prob, num_same / len(results)))
        assert abs(num_same / len(results) - mpe_prob) <= 1e-2
    print("** Testing MPE and Random (res=%d)" % res)
    test_round(octree_belief)
    octree_belief[ObjectState(1, "cube", (0,0,1), res=1)] = TEST_ALPHA
    test_round(octree_belief)
    octree_belief[ObjectState(1, "cube", (0,0,1), res=1)] = DEFAULT_VAL

def test_time(octree_belief):
    """Try setting cells to be BETA and see if it affects the
    likelihood of sampling from the remaining cells."""
    start = time.time()
    for i in range(1000):
        octree_belief.random(res=1)
    tot_res1 = time.time() - start

    start = time.time()
    for i in range(1000):
        octree_belief.random(res=2)
    tot_res2 = time.time() - start

    start = time.time()
    for i in range(1000):
        octree_belief.random(res=4)
    tot_res4 = time.time() - start
    print("Avg sample time (res=1): %.3f" % (tot_res1 % 1000))
    print("Avg sample time (res=2): %.3f" % (tot_res2 % 1000))
    print("Avg sample time (res=4): %.3f" % (tot_res4 % 1000))


def test_visualize(octree_belief):
    fig = plt.gcf()
    ax = fig.add_subplot(1,1,1,projection="3d")
    m = plot_octree_belief(ax, octree_belief,
                           alpha="clarity", edgecolor="black", linewidth=0.1)
    ax.set_xlim([0, octree_belief.octree.dimensions[0]])
    ax.set_ylim([0, octree_belief.octree.dimensions[1]])
    ax.set_zlim([0, octree_belief.octree.dimensions[2]])
    ax.grid(False)
    fig.colorbar(m, ax=ax)
    plt.show(block=False)
    plt.pause(1)
    ax.clear()


def test_belief_update(octree_belief):
    print("** Testing belief update")

    # We will make a camera, and place it at a pose.
    camera = FrustumCamera(fov=60, aspect_ratio=0.85, near=1, far=5)
    robot_pose = (3, 4, 5, 90, 0, 0)

    objid = octree_belief.objid
    voxels = {(x,y,z): Voxel((x,y,z), Voxel.FREE)
              for (x,y,z) in camera.get_volume(robot_pose)}
    voxels[(2, 2, 2)] = Voxel((2,2,2), objid)
    voxels_obz = FovVoxels(voxels)

    octree_belief = update_octree_belief(
        octree_belief, voxels_obz,
        alpha=TEST_ALPHA, beta=TEST_BETA)  # this setting is for log space

    test_visualize(octree_belief)


def test_belief_update_big_voxel(octree_belief):
    print("** Testing belief update with big voxel")

    # We will make a camera, and place it at a pose.
    camera = FrustumCamera(fov=60, aspect_ratio=0.85, near=1, far=5)
    robot_pose = (3, 4, 5, 90, 0, 0)

    objid = octree_belief.objid
    voxels = {(x//2,y//2,z//2,2): Voxel((x//2,y//2,z//2,2), Voxel.FREE)
              for (x,y,z) in camera.get_volume(robot_pose)}
    voxels[(1, 4, 1, 2)] = Voxel((1,4,1,2), objid)
    voxels_obz = FovVoxels(voxels)

    octree_belief = update_octree_belief(
        octree_belief, voxels_obz,
        alpha=TEST_ALPHA, beta=TEST_BETA)  # this setting is for log space

    test_visualize(octree_belief)
