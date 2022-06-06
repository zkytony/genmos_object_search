import random
import pomdp_py
import config_test1 as test_config
from sloop.osm.datasets import MapInfoDataset
from sloop_object_search.utils.misc import import_class
from sloop_object_search.utils.osm import osm_map_to_grid_map
from sloop_object_search.oopomdp.agent import SloopMosBasic2DAgent, VizSloopMosBasic2D


def main(config):
    mapinfo = MapInfoDataset()
    map_name = config['task_config']["map_name"]
    mapinfo.load_map(map_name)
    grid_map = osm_map_to_grid_map(map_info, map_name)

    init_robot_pose = (*random.sample(grid_map.free_locations, 1)[0], 0.0)
    config["agent_config"]["robot"]["init_pose"] = init_robot_pose
    init_robot_state = RobotState(robot["id"], robot["init_pose"], tuple(), None)
    objects = config["agent_config"]["objects"]
    object_states = {
        objid: ObjectState(objid, objects[objid]["class"],
                           random.sample(grid_map.free_locations - {init_robot_pose[:2]}, 1))
        for objid in objects["targets"]
    }

    init_state = pomdp_py.OOState({**{robot["id"]: init_robot_state},
                                   **object_states})
    agent = eval(config["agent_config"]["agent_class"])(config["agent_config"])
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)
    planner = import_class(config["planner_config"])(**config["planner_params"],
                                                     rollout_policy=agent.policy_model)
    max_steps = config["task_config"]["max_steps"]
    visualizer = VizSloopMosBasic2D(grid_map)
    img = visualizer.render()
    visualizer.show_img(img)

    for i in range(max_steps):
        action = planner.plan(agent)
        reward = task_env.state_transition(action, execute=True)
        observation = task_env.provide_observation(agent.observation_model)
        agent.update_belief(observation, action)
        img = visualizer.render()
        visualizer.show_img(img)

if __name__ == "__main__":
    main(test_config)
