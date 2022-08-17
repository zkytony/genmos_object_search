import os
import pomdp_py
import pytest
import random
from sloop.osm.datasets.map_info_utils import register_map, load_filepaths
from sloop_object_search.oopomdp.domain.state import RobotState
from sloop_object_search.oopomdp.agent.basic3d import MosAgentBasic3D
from sloop_object_search.oopomdp.models.octree_belief import Octree, OctreeBelief
from sloop_object_search.oopomdp.models.search_region import SearchRegion3D

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def config():
    config = {
        "agent_config": {
            "no_look": True,
            "belief": {},
            "targets": ["G"],
            "objects": {
                "G": {
                    "class": "car",
                    "transition": {
                        "class": "sloop_object_search.oopomdp.StaticObjectTransitionModel"
                    },
                    "color": [100, 200, 80]
                },
            },
            "robot": {
                "id": "robot0",
                "detectors": {
                    "G": {
                        "class": "sloop_object_search.oopomdp.FrustumVoxelAlphaBeta",
                        "params": (dict(fov=60, near=0.1, far=5), (1e5, 0.1)),
                    },
                },
                "action": {
                    "func": "sloop_object_search.oopomdp.domain.action.basic_discrete_moves3d",
                    "params": {
                        "step_size": 1,
                        "rotation": 90.0,
                        "scheme": "axis"
                    }
                }
            },
        },
    }
    return config

@pytest.fixture
def search_region():
    octree = Octree(None, (16, 16, 16), default_val=0)
    for i in range(20):
        x = random.randint(0, 15)
        y = random.randint(0, 15)
        z = random.randint(0, 15)
        octree.add_node(x,y,z,1, val=1)
    search_region = SearchRegion3D(octree, search_space_resolution=0.5)
    return search_region


def test_basic3d_agent_basics(config, search_region):
    init_robot_pose_dist =  pomdp_py.Histogram({(0,-1,4,0,0,0,1): 1.0})
    agent = MosAgentBasic3D(config["agent_config"], search_region, init_robot_pose_dist)
    pouct = pomdp_py.POUCT(planning_time=0.5, rollout_policy=agent.policy_model)
    print(pouct.plan(agent))
