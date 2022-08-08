# Used to specifically test MOS3D 3D Object Search agent.
import copy
import random
import pomdp_py
from pomdp_py.utils import typ

from sloop_object_search.utils.misc import import_class
from sloop_object_search.utils.math import normalize, euler_to_quat
from sloop_object_search.oopomdp.planner import make_planner
from sloop_object_search.oopomdp.agent.basic3d import MosBasic3DAgent


def main(_config):

    # TODO: hard coded initial pose
    init_robot_pose = (2, -2, 4, *euler_to_quat(0, 0, 0))
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_robot_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"])

    _planner_config = _config["planner_config"]
    planner = make_planner(_planner_config, agent)
    max_steps = _config["task_config"]["max_steps"]

    # Just grab a random state as initial state
    random.seed(100)
    init_state = agent.belief.random()
    task_env = pomdp_py.Environment(init_state,
                                    agent.transition_model,
                                    agent.reward_model)

    # Show visualization
    _task_config = _config["task_config"]
    # TODO: visualization

    # Belief prior
    _objects = _config["agent_config"]["objects"]

    for objid in agent.belief.object_beliefs:
        _prior = _config["agent_config"]["belief"]["prior"]
        _prior_obj = _prior.get(objid, {})
        if _prior_obj == "groundtruth":
            true_obj_state = task_env.state.s(objid)
            belief_obj = {si: 1e-9 for si in agent.belief.b(objid)}
            belief_obj[true_obj_state] = 1.0
            agent.belief.set_object_belief(objid, pomdp_py.Histogram(normalize(belief_obj)))

    for i in range(max_steps):
        action = planner.plan(agent)
        if hasattr(agent, "tree") and agent.tree is not None:
            _dd = pomdp_py.utils.TreeDebugger(agent.tree)
            _dd.p(1)

        reward = task_env.state_transition(action, execute=True)
        observation = task_env.provide_observation(agent.observation_model, action)
        print("Step {}:  Action: {}   Observation: {}  Reward: {}    Robot State: {}"\
              .format(i, action, observation, reward, task_env.state.s(agent.robot_id)))

        agent.update_belief(observation, action)
        planner.update(agent, action, observation)
        if isinstance(agent, SloopMosTopo2DAgent):
            task_env.state.set_object_state(agent.robot_id, agent.belief.mpe().s(agent.robot_id))
        visualize_step(viz, agent, task_env, action, _config)

        if set(task_env.state.s(agent.robot_id).objects_found)\
           == set(_config["agent_config"]["targets"]):
            print("Done.")
            break

if __name__ == "__main__":
    import sys
    import importlib
    if len(sys.argv) >= 2:
        config_module = importlib.import_module(sys.argv[1].split(".py")[0])
    else:
        import config_test_basic2d
        config_module = config_test_basic2d
    main(config_module.config)
