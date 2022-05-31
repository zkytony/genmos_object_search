# Tests the MosOOPOMDP in problem.py; purely object search on grid world
import random
import pomdp_py
import time
from sloop.oopomdp.problem import MosOOPOMDP, MosViz
from sloop.oopomdp.example_worlds import random_world
from sloop.oopomdp.env.env import (make_laser_sensor,
                                   make_proximity_sensor)

### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def setup_solve(problem,
                max_depth=10,  # planning horizon
                discount_factor=0.99,
                planning_time=-1.,       # amount of time (s) to plan each step
                num_sims=-1,
                exploration_const=1000, # exploration constant
                visualize=True,
                bg_path=None):
    """Set up the components before pomdp loop starts.
    So that the results of this setup (e.g. the visualizer)
    can be reused else where."""
    random_objid = random.sample(problem.env.target_objects, 1)[0]
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 num_sims=num_sims,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(random_object_belief, pomdp_py.Particles):
        # Use POMCP
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 num_sims=num_sims,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))

    robot_id = problem.agent.robot_id
    viz = None
    if visualize:
        # controllable=False means no keyboard control.
        viz = MosViz(problem.env, controllable=False, bg_path=bg_path, res=15)
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   problem.agent.cur_belief)
        viz.on_render()

    return planner, viz


def solve(problem, planner,
          viz=None,
          step_func=None, # makes this function more extensible.
          step_func_args={},
          max_time=120,  # maximum amount of time allowed to solve the problem
          max_steps=500): # maximum number of planning steps the agent can take.):
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.
    """
    robot_id = problem.agent.robot_id

    _time_used = 0
    _find_actions_count = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        # Plan action
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break  # no more time to update.

        # Execute action
        reward = problem.env.state_transition(real_action, execute=True,
                                              robot_id=robot_id)

        # Receive observation
        _start = time.time()
        real_observation = \
            problem.env.provide_observation(problem.agent.observation_model, real_action)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        belief_update(problem.agent, real_action, real_observation,
                      problem.env.state.object_states[robot_id],
                      planner)
        _time_used += time.time() - _start

        # Info and render
        _total_reward += reward
        if isinstance(real_action, FindAction):
            _find_actions_count += 1
        print("==== Step %d ====" % (i+1))
        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        print("Find Actions Count: %d" %  _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__planning_time__: %.5f" % planner.last_planning_time)

        if viz is not None:
            # This is used to show the sensing range; Not sampled
            # according to observation model.
            robot_pose = problem.env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            if isinstance(real_action, LookAction) or isinstance(real_action, FindAction):
                viz_observation = \
                    problem.env.sensors[robot_id].observe(robot_pose,
                                                          problem.env.state)
            viz.update(robot_id,
                       real_action,
                       real_observation,
                       viz_observation,
                       problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()

        if step_func is not None:
            step_func(problem, real_action, real_observation, reward, viz,
                      **step_func_args)

        # Termination check
        if set(problem.env.state.object_states[robot_id].objects_found)\
           == problem.env.target_objects:
            print("Done!")
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break
    return _total_reward

# Test
def unittest(bg_path=None):
    # random world
    grid_map, robot_char = random_world(20, 20, 20, 10)
    laserstr = make_laser_sensor(90, (1, 3), 0.5, False)
    proxstr = make_proximity_sensor(5, False)
    problem = MosOOPOMDP(robot_char,  # r is the robot character
                         sigma=0.01,  # observation model parameter
                         epsilon=1.0, # observation model parameter
                         grid_map=grid_map,
                         sensors={robot_char: laserstr},
                         prior="uniform",
                         no_look=True,
                         reward_small=10,
                         agent_has_map=True,
                         dimension=(10,10))
    planner, viz = setup_solve(problem,
                               max_depth=20,  # planning horizon
                               discount_factor=0.95,
                               num_sims=500,  # Number of simulations per step
                               exploration_const=1000, # exploration constant
                               visualize=True,
                               bg_path=bg_path)
    print(problem.env.state)
    solve(problem, planner, viz=viz,
          max_time=1e9,
          max_steps=1000)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        bg_path = sys.argv[1]
    else:
        bg_path = None
    unittest(bg_path=bg_path)
