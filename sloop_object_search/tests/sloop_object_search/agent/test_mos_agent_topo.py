import pomdp_py
import pytest
from test_mos_agent_basic import config, search_region
from sloop_object_search.oopomdp.agent.topo2d import MosAgentTopo2D

def test_topo2d_agent_basics(config, search_region):
    init_robot_pose_dist =  pomdp_py.Histogram({(0,-1,4): 1.0})
    search_region.grid_map.label_all(
        search_region.grid_map.free_locations, "topo_position_candidate")
    agent = MosAgentTopo2D(config["agent_config"], search_region, init_robot_pose_dist)
    pouct = pomdp_py.POUCT(planning_time=0.5, rollout_policy=agent.policy_model)
    print(pouct.plan(agent))
