import os
import pomdp_py
import pytest
from test_mos_agent_basic import config, search_region
from sloop.osm.datasets.map_info_utils import register_map, load_filepaths
from sloop_object_search.oopomdp.agent.topo2d import MosAgentTopo2D, SloopMosAgentTopo2D

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

def test_topo2d_agent_basics(config, search_region):
    init_robot_pose_dist =  pomdp_py.Histogram({(0,-1,4): 1.0})
    search_region.grid_map.label_all(
        search_region.grid_map.free_locations, "topo_position_candidate")
    agent = MosAgentTopo2D(config["agent_config"], search_region, init_robot_pose_dist)
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
    init_robot_pose_dist =  pomdp_py.Histogram({(0,-1,4): 1.0})
    try:
        register_map(search_region.grid_map,
                     search_region.grid_size,
                     "./data")
    except FileExistsError:
        print(f"map {search_region.grid_map} is already registered.")
        load_filepaths(search_region.grid_map.name,
                       search_region.grid_size,
                       "./data")

    search_region.grid_map.label_all(
        search_region.grid_map.free_locations, "topo_position_candidate")
    agent = SloopMosAgentTopo2D(config["agent_config"], search_region, init_robot_pose_dist)
    pouct = pomdp_py.POUCT(planning_time=0.5, rollout_policy=agent.policy_model)
    print(pouct.plan(agent))
