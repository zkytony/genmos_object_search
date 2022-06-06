import random
import pomdp_py
import config_test1 as test_config
from sloop.osm.datasets import MapInfoDataset, FILEPATHS
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

    # Just grab a random state as initial state
    init_state = agent.belief.random()
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)
    _planner_config = _config["planner_config"]
    planner = import_class(_planner_config["planner"])(**_planner_config["planner_params"],
                             rollout_policy=agent.policy_model)
    max_steps = _config["task_config"]["max_steps"]
    visualizer = VizSloopMosBasic2D(agent.grid_map,
                                    res=20,
                                    bg_path=FILEPATHS[map_name]["map_png"])
    img = visualizer.render()
    visualizer.show_img(img)

    for i in range(max_steps):
        action = planner.plan(agent)
        _dd = pomdp_py.utils.TreeDebugger(agent.tree)
        _dd.p(1)

        reward = task_env.state_transition(action, execute=True)
        observation = task_env.provide_observation(agent.observation_model, action)
        agent.update_belief(observation, action)
        planner.update(agent, action, observation)
        img = visualizer.render()
        visualizer.show_img(img)

if __name__ == "__main__":
    main(test_config.config)
