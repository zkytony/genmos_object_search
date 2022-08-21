import time
import random
from sloop_object_search.oopomdp import ObjectState
from sloop_object_search.oopomdp.models.octree_belief import Octree, OctreeBelief
from sloop_object_search.utils.misc import timeout


def test_octree_belief_performance_small():
    print("\ntest_octree_belief_performance_SMALL")
    with timeout(45):
        _test_octree_belief_performance((16, 16, 16))

def test_octree_belief_performance_normal():
    print("\ntest_octree_belief_performance_NORMAL")
    with timeout(45):
        _test_octree_belief_performance((32, 32, 32))

def test_octree_belief_performance_medium():
    print("\ntest_octree_belief_performance_MEDIUM")
    with timeout(45):
        _test_octree_belief_performance((64, 64, 64))

def test_octree_belief_performance_big():
    print("\ntest_octree_belief_performance_BIG")
    with timeout(45):
        _test_octree_belief_performance((128, 128, 128))

def test_octree_belief_performance_large():
    print("\ntest_octree_belief_performance_LARGE")
    with timeout(45):
        _test_octree_belief_performance((256, 256, 256))

def test_octree_belief_performance_huge():
    print("\ntest_octree_belief_performance_HUGE")
    with timeout(45):
        _test_octree_belief_performance((1024, 1024, 1024))


def _test_octree_belief_performance(dim,
                                    sample_count=2000):
    octree = Octree(dim)
    octree_belief = OctreeBelief(*dim, 1, "cube", octree)

    # Sampling from the tree - initial state
    print("\n------ sampling from initial tree ------")
    _start_time = time.time()
    for i in range(sample_count):
        s = octree_belief.random()
    _total_time = time.time() - _start_time
    num_leaves = len(octree_belief.octree.get_leaves())
    print(f"Sampling {dim} octree belief (with {num_leaves} leaves) {sample_count} took {_total_time:.9f}s) ({_total_time/sample_count:.9f}s per sample)\n")

    # Sampling from partial tree
    print("------ sampling from partial tree ------")
    num_samples = int(dim[0]*dim[1]*dim[2]*0.5)
    positions = random.sample({(x,y,z)   # random sample without replacement
                               for x in range(dim[0])
                               for y in range(dim[1])
                               for z in range(dim[2])}, num_samples)
    _start_time = time.time()
    for p in positions:
        si = ObjectState(1, "cube", p, res=1)
        octree_belief[si] = random.uniform(0,1)*10
    _total_time = time.time() - _start_time
    print(f"Inserting {num_samples} leaves to octree belief took {_total_time:.4f}s ({_total_time/len(positions)}:.9f)s per leaf")

    _start_time = time.time()
    for i in range(sample_count):
        s = octree_belief.random()
    _total_time = time.time() - _start_time
    num_leaves = len(octree_belief.octree.get_leaves())
    print(f"Sampling {dim} octree belief (with {num_leaves} leaves) {sample_count} took {_total_time:.4f}s) ({_total_time/sample_count:.9f}s per sample)\n")

    # Sampling from partial tree
    print("------ sampling from full tree ------")
    # reset the tree.
    octree = Octree(dim)
    octree_belief = OctreeBelief(*dim, 1, "cube", octree)
    num_samples = int(dim[0]*dim[1]*dim[2])
    positions = random.sample({(x,y,z)   # random sample without replacement
                               for x in range(dim[0])
                               for y in range(dim[1])
                               for z in range(dim[2])}, num_samples)
    _start_time = time.time()
    for p in positions:
        si = ObjectState(1, "cube", p, res=1)
        octree_belief[si] = random.uniform(0,1)*10
    _total_time = time.time() - _start_time
    print(f"Inserting {num_samples} leaves to octree belief took {_total_time:.4f}s ({_total_time/len(positions)}:.9f)s per leaf")

    _start_time = time.time()
    for i in range(sample_count):
        s = octree_belief.random()
    _total_time = time.time() - _start_time
    num_leaves = len(octree_belief.octree.get_leaves())
    print(f"Sampling {dim} octree belief (with {num_leaves} leaves) {sample_count} took {_total_time:.4f}s) ({_total_time/sample_count:.9f}s per sample)\n\n")
