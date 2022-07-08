import copy
import random
import pomdp_py
from pomdp_py.utils import typ
import sloop.observation
from sloop.osm.datasets import MapInfoDataset, FILEPATHS
from sloop_object_search.utils.misc import import_class
from sloop_object_search.utils.math import normalize
from sloop_object_search.oopomdp.agent import (SloopMosBasic2DAgent,
                                               VizSloopMosBasic2D,
                                               SloopMosTopo2DAgent)
from sloop_object_search.oopomdp.domain.state import RobotState, ObjectState, RobotStateTopo
from sloop_object_search.oopomdp.domain.action import LookAction
from sloop_object_search.oopomdp.planner import make_planner
from sloop_object_search.oopomdp.planner.hier2d import HierarchicalPlanner


def ask_for_splang(sloop_agent, objspec=None):
    _help_msg = "Please describe the location"
    if objspec is None:
        _help_msg += " of object(s) on the map using natural language:"
    else:
        _help_msg += f" of {objspec['class']} using natural language:"
    print(typ.bold(_help_msg))
    splang = input(typ.green("  Your input: "))
    splang_obz = sloop.observation.parse(splang,
                                         sloop_agent.map_name,
                                         kwfile=FILEPATHS["symbol_to_synonyms"],
                                         spacy_model=sloop_agent.spacy_model,
                                         verbose_level=1)
    return splang_obz

def visualize_step(viz, agent, env, action, _config, **kwargs):
    objlocs = {j: env.state.s(j).loc
               for j in env.state.object_states
               if j != agent.robot_id}
    colors = {j: _config["agent_config"]["objects"][j].get("color", [128, 128, 128])
              for j in env.state.object_states
              if j != agent.robot_id}
    no_look = _config["agent_config"]["no_look"]

    draw_fov = list(objlocs.keys())
    if not no_look:
        if not isinstance(action, LookAction):
            draw_fov = None
    if action is None:
        draw_fov = None
    img = viz.render(agent, objlocs, colors=colors, draw_fov=draw_fov, **kwargs)
    img = viz.highlight(img, [[0, 2]], color=(80, 80, 230))
    viz.show_img(img)

def main(_config):
    map_name = _config['task_config']["map_name"]

    init_robot_pose = (5, 10, 0.0)
    _robot = _config["agent_config"]["robot"]
    _robot["init_pose"] = init_robot_pose
    agent = eval(_config["agent_config"]["agent_class"])(
        _config["agent_config"], map_name)

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
    viz = import_class(_task_config["visualizer"])(agent.grid_map,
                                                   bg_path=FILEPATHS[map_name]["map_png"],
                                                   **_task_config["viz_params"])
    _prior = _config["agent_config"]["belief"]["prior"]
    if _prior == "splang" or any(_prior[objid] == "splang" for objid in _prior):
        draw_belief = False  # don't hide the map when typing language
    else:
        draw_belief = True
    visualize_step(viz, agent, task_env, None, _config, draw_belief=draw_belief)

    # Belief prior
    _objects = _config["agent_config"]["objects"]
    if _prior == "splang":
        splang_observation = ask_for_splang(agent)
        agent.update_belief(splang_observation, None)

    for objid in agent.belief.object_beliefs:
        _prior_obj = _prior.get(objid, {})
        if _prior_obj == "groundtruth":
            true_obj_state = task_env.state.s(objid)
            belief_obj = {si: 1e-9 for si in agent.belief.b(objid)}
            belief_obj[true_obj_state] = 1.0
            agent.belief.set_object_belief(objid, pomdp_py.Histogram(normalize(belief_obj)))

        elif _prior_obj == "splang":
            splang_observation = ask_for_splang(agent, _objects[objid])
            agent.update_belief(splang_observation, None)

    for i in range(max_steps):
        action = planner.plan(agent)
        # if hasattr(agent, "tree") and agent.tree is not None:
        #     _dd = pomdp_py.utils.TreeDebugger(agent.tree)
        #     _dd.p(1)

        if isinstance(planner, HierarchicalPlanner):
            # note: planner-specific (but ok since this is just a test!)
            old_state = copy.deepcopy(task_env.state)
            snext = planner.mos2d_agent.transition_model.sample(task_env.state, action)
            srobot_next = snext.s(agent.robot_id)
            srobot_topo_next = RobotStateTopo(agent.robot_id,
                                              srobot_next['pose'],
                                              srobot_next['objects_found'],
                                              srobot_next['camera_direction'],
                                              task_env.state.s(srobot_next['id']).topo_nid)
            snext.set_object_state(agent.robot_id, srobot_topo_next)
            task_env.apply_transition(snext)
            reward = agent.reward_model.sample(old_state, action, snext)
        else:
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
