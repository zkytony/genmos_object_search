import os
import pomdp_py
import pytest
from sloop.osm.datasets.map_info_utils import register_map, load_filepaths
from sloop_object_search.oopomdp.domain.state import RobotState
from sloop_object_search.oopomdp.agent.basic2d import MosAgentBasic2D, SloopMosAgentBasic2D
from sloop_object_search.oopomdp.models.grid_map2 import GridMap2
from sloop_object_search.oopomdp.models.search_region import SearchRegion2D

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
                        "class": "sloop_object_search.oopomdp.FanModelSimpleFP",
                        "params": (dict(fov=90, min_range=0, max_range=5), (0.9, 0.1, 0.25))
                    },
                },
                "action": {
                    "func": "sloop_object_search.oopomdp.domain.action.basic_discrete_moves2d",
                    "params": {
                        "step_size": 3,
                        "h_rotation": 45.0
                    }
                }
            },
        },
    }
    return config

@pytest.fixture
def search_region():
    free_locations = {(x,y)
                      for x in range(2,8)
                      for y in range(2,6)}
    obstacles = {(x,y)
                 for x in range(-5,1)
                 for y in range(-1,2)}
    grid_map2 = GridMap2(obstacles=obstacles,
                         free_locations=free_locations)
    search_region = SearchRegion2D(grid_map2, grid_size=0.5)
    return search_region


def test_basic2d_agent_basics(config, search_region):
    init_robot_belief =  pomdp_py.Histogram(
        {RobotState("robot", (0,-1,4), (), None) : 1.0}
    )
    agent = MosAgentBasic2D(config["agent_config"], search_region, init_robot_belief)
    pouct = pomdp_py.POUCT(planning_time=0.5, rollout_policy=agent.policy_model)
    print(pouct.plan(agent))

def test_basic2d_sloop_agent_basics(config, search_region):
    config["agent_config"].update({
        "spacy_model": "en_core_web_lg",
        'foref_model_map_name': 'honolulu',
        "foref_models_dir": os.path.join(ABS_PATH, "../../../models"),
        "object_symbol_map": {
            # Maps from object symbol to object id
            "GreenToyota": "G"
        },
    })
    init_robot_belief =  pomdp_py.Histogram(
        {RobotState("robot", (0,-1,4), (), None) : 1.0}
    )
    try:
        register_map(search_region.grid_map,
                     search_region.grid_size,
                     "./data")
    except FileExistsError:
        print(f"map {search_region.grid_map} is already registered.")
        load_filepaths(search_region.grid_map.name,
                       search_region.grid_size,
                       "./data")

    agent = SloopMosAgentBasic2D(config["agent_config"], search_region, init_robot_belief)
    pouct = pomdp_py.POUCT(planning_time=0.5, rollout_policy=agent.policy_model)
    print(pouct.plan(agent))
