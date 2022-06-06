import random
import pomdp_py
import config_test1 as test_config
from sloop.osm.datasets import MapInfoDataset
from sloop_object_search.utils.misc import import_class
from sloop_object_search.oopomdp.agent import SloopMosBasic2DAgent, VizSloopMosBasic2D
from sloop_object_search.oopomdp.domain.state import RobotState, ObjectState


def main(_config):
    map_name = _config['task_config']["map_name"]

    init_robot_pose = (0, 0, 0.0)
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_robot_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"], map_name)

    init_robot_state = agent.belief.mpe().s(_robot["id"])
    _objects = _config["agent_config"]["objects"]
    object_states = {
        objid: ObjectState(objid, _objects[objid]["class"],
                           random.sample(grid_map.free_locations - {init_robot_pose[:2]}, 1))
        for objid in _objects["targets"]
    }
    init_state = pomdp_py.OOState({**{_robot["id"]: init_robot_state},
                                   **object_states})
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)
    planner = import_class(config["planner_config"])(**_config["planner_params"],
                                                     rollout_policy=agent.policy_model)
    max_steps = _config["task_config"]["max_steps"]
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
    main(test_config.config)
