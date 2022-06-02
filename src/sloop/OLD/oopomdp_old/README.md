# Multi-Object Search OOPOMDP for OSM

This code builds upon [the MOS example in pomdp_py](https://h2r.github.io/pomdp-py/html/examples.mos.html).

Requires installation of [pomdp_py](https://h2r.github.io/pomdp-py/html/installation.html).

**Note**
The step size is changed to 2 already. To integrate with Unity I guess you only need to copy over the problem_osm.py file (or related files) that lets it talk with ROS and Unity.






### Run this on ROS

Copy and paste `problem.py` into `problem_ros.py`.

Add the following lines after `MosOOPOMDP` class definition.
```python
### ROS Setup ###
drone_reached = False # This will tell us whether to send next action or not
skydio_observation = MosOOObservation({})

def skydiostring_callback(msg):
    global drone_reached

    if msg.data == "Waiting...":
        drone_reached = True
    else:
        drone_reached = False

def observation_callback(msg):
    global skydio_observation

    obs = msg.data

    # Convert string to dictionary
    obs_dict = ast.literal_eval(obs)
    # print("obs_dict: ", obs_dict)
    # Convert None to ObjectObservation.NULL
    for key, value in obs_dict.items():
        if value is None:
            obs_dict[key] = ObjectObservation.NULL
        else:
            # convert lat/lon to pomdp grid coords
            str_tuple = latlon_to_pomdp_cell(value[0], value[1], pomdp_to_map_fp, idx_to_cell_fp)
            # obs_dict[key] = tuple(reversed(ast.literal_eval(str_tuple)))
            obs_dict[key] = ast.literal_eval(str_tuple)
            # obs_dict[key] = ast.literal_eval(value)
    if not obs_dict[1] == None:
        obs_dict[1] = (3, 13)

    observation = MosOOObservation(obs_dict)
    skydio_observation = observation

rospy.init_node("spatial_lang")
rospy.Subscriber("/skydiostring", String, skydiostring_callback)
rospy.Subscriber("/observation", String, observation_callback)
gpose_pub = rospy.Publisher("/gpose", PoseStamped, queue_size=10)
rate = rospy.Rate(10)
```

Then, replace the `solve()` function by the following
```python
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
    global drone_reached
    global skydio_observation

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
            print("Time used: ", _time_used)
            print("Total reward: ", _total_reward)
            print("Maximum time reached.")
            break  # no more time to update.

        robot_pose = problem.env.state.object_states[robot_id].pose
        if real_action.name != "find":
            action_pose = real_action.motion
            next_coordinate = (robot_pose[0] + action_pose[0], robot_pose[1] + action_pose[1])
            next_latlon = get_center_latlon(next_coordinate, pomdp_to_map_fp, idx_to_cell_fp)

            # Publish lat/lon pair to Unity sim on /gpose
            p = PoseStamped()
            # formatting for Unity
            p.pose.position.x = next_latlon[1]
            p.pose.position.y = -1 * next_latlon[0]
            p.pose.position.z = 0.0
            gpose_pub.publish(p)
            rospy.sleep(2)

        # Execute action
        reward = problem.env.state_transition(real_action, execute=True,
                                              robot_id=robot_id)

        # Receive observation
        while not drone_reached:
            rospy.sleep(0.25)

        if drone_reached:
            real_observation = skydio_observation
        if real_action.name == "find":
            print("find time: ", _time_used)
            real_observation = MosOOObservation({})

        # _start = time.time()
        pomdp_observation = \
            problem.env.provide_observation(problem.agent.observation_model, real_action)

        print("pomdp observation: ", pomdp_observation)
        print("skydio observation: ", real_observation)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)

        _start = time.time()
        belief_update(problem.agent, real_action, real_observation,
                      problem.env.state.object_states[robot_id],
                      planner)
        # _time_used += time.time() - _start

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

        if viz is not None:
            # This is used to show the sensing range; Not sampled
            # according to observation model.
            robot_pose = problem.env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            # TODO: Is this correct when on no_look?
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
            print("Time used: ", _time_used)
            print("Total reward: ", _total_reward)
            print("Done!")
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("Time used: ", _time_used)
            print("Total reward: ", _total_reward)
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Time used: ", _time_used)
            print("Total reward: ", _total_reward)
            print("Maximum time reached.")
            break
```
There may be some bugs since this code may be out-of-date, but the above tells you the gist of how to run this on ROS.
