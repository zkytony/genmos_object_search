# Test conversion of belief between 2D and 3D
import pickle
import open3d as o3d
import pytest
import random
from sloop_object_search.oopomdp.models.belief\
    import object_belief_2d_to_3d, update_2d_belief_by_3d, init_object_beliefs_2d
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.oopomdp.models.search_region import SearchRegion2D, SearchRegion3D
from sloop_object_search.oopomdp.models.octree_belief import OccupancyOctreeDistribution
from sloop_object_search.utils import open3d_utils
from sloop_object_search.utils.colors import cmaps

@pytest.fixture
def search_region2d():
    with open("rsc/search_region2d_test.pkl", "rb") as f:
        search_region2d = pickle.load(f)
    return search_region2d

@pytest.fixture
def bobj2d(search_region2d):
    with open("rsc/bobj2d_test.pkl", "rb") as f:
        bobj2d = pickle.load(f)
    return bobj2d

@pytest.fixture
def search_region3d():
    with open("rsc/search_region3d_test.pkl", "rb") as f:
        search_region3d = pickle.load(f)
    return search_region3d


def test_belief_conversion(bobj2d, search_region2d, search_region3d):
    bobj3d = object_belief_2d_to_3d(bobj2d, search_region2d, search_region3d, res=8)
    print("2D->3D")
    geometries = open3d_utils.draw_search_region3d(
        search_region3d, octree_dist=bobj3d.octree_dist, viz=False,
        cmap=cmaps.COLOR_MAP_GRAYS)
    geometries.extend(open3d_utils.draw_locdist2d(
        bobj2d.loc_dist, search_region=search_region2d, viz=False))
    o3d.visualization.draw_geometries(geometries)

    print("3D->2D")
    bobj2d_projected = update_2d_belief_by_3d(
        bobj2d, bobj3d, bobj3d.octree_dist.normalizer*2.0, search_region2d, search_region3d)
    geometries = open3d_utils.draw_search_region3d(
        search_region3d, octree_dist=bobj3d.octree_dist, viz=False,
        cmap=cmaps.COLOR_MAP_GRAYS)
    geometries.extend(open3d_utils.draw_locdist2d(
        bobj2d_projected.loc_dist, search_region=search_region2d, viz=False))
    o3d.visualization.draw_geometries(geometries)
