from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
from sloop_object_search.oopomdp.experiments.result_types import *
import sloop_object_search.oopomdp.problem as mos
from sloop_object_search.oopomdp.experiments.constants import *
from sloop.osm.datasets.utils import euclidean_dist
from sloop.osm.datasets import FILEPATHS
import random
import pomdp_py
import time
import copy


def get_id(obj_symbol):
    return obj_id_map[obj_letter_map[obj_symbol]]

def make_trial(trial_name, worldstr, map_name, language, sensor, prior_type,
               prior, prior_metadata, model_name, **kwargs):
    problem_args = {"sigma": kwargs.get("sigma", 0.01),
                    "epsilon": kwargs.get("epsilon", 1.0),
                    "agent_has_map": kwargs.get("agent_has_map", True),
                    "reward_small": kwargs.get("small", 10),
                    "sensors": {"r": sensor},
                    "no_look": kwargs.get("no_look", True)}
    solver_args = {"max_depth": kwargs.get("max_depth", 30),
                   "discount_factor": kwargs.get("discount_factor", 0.95),
                   "planning_time": kwargs.get("planning_time", -1.),
                   "num_sims": kwargs.get("num_sims", -1),
                   "exploration_const": kwargs.get("exploration_const", 1000)}
    exec_args = {"max_time": kwargs.get("max_time", 360),
                 "max_steps": kwargs.get("max_steps", 200),
                 "visualize": kwargs.get("visualize", False)}
    sensor_str = sensor.replace(" ", ":").replace("_", "*")
    return SloopPriorTrial("%s_%s-%s-%s-%s" % (trial_name, prior_type,
                                               map_name.replace("_",","),
                                               sensor_str, model_name.replace("_",">")),
                           config={"problem_args": problem_args,
                                   "solver_args": solver_args,
                                   "exec_args": exec_args,
                                   "world": worldstr,
                                   "map_name": map_name,
                                   "prior_type": prior_type,
                                   "prior": prior,
                                   "prior_metadata": prior_metadata,
                                   "language": language,
                                   "obj_id_map": kwargs.get("obj_id_map", {})})

def create_world(width, length, robot_pose, landmark_poses, target_poses, target_objects):
    worldstr = [[ "." for i in range(width)] for j in range(length)]

    rx, ry = robot_pose
    worldstr[ry][rx] = "r"

    for i, pose in enumerate(target_poses):
        x, y = pose
        assert worldstr[y][x] == ".", "Invalid landmark pose %s" % str((x,y))
        worldstr[y][x] = target_objects[i]  # target_objects[i] is a character

    for x,y in landmark_poses:
        assert worldstr[y][x] == ".", "Invalid landmark pose %s" % str((x,y))
        worldstr[y][x] = "x"

    # Create the string.
    finalstr = []
    for row_chars in worldstr:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)
    return finalstr


def get_prior(prior_type, model, query, map_name, mapinfo,
              **kwargs):

    metadata = {}
    if prior_type == "uniform":
        prior = "uniform"

    elif prior_type == "informed":
        prior = "informed"

    elif prior_type.startswith("informed"):
        obj_poses = kwargs.get("obj_poses")
        prior = model.interpret(obj_poses, map_name, mapinfo)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}

    elif prior_type.startswith("keyword"):
        prior, meta = model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    elif prior_type.startswith("rule"):
        prior, meta =\
            model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    elif prior_type.startswith("mixture"):
        prior, meta =\
            model.interpret(query, map_name, mapinfo, **kwargs)
        prior = {get_id(symbol):prior[symbol]
                 for symbol in prior}
        metadata.update(meta)

    return prior, metadata



class SloopPriorTrial(Trial):
    RESULT_TYPES = [RewardsResult, StatesResult, PriorQualityResult, LangResult]
    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)


    def _setup_solve(self, problem,
                     max_depth=10,  # planning horizon
                     discount_factor=0.99,
                     planning_time=-1.,       # amount of time (s) to plan each step
                     num_sims=-1,
                     exploration_const=1000, # exploration constant
                     visualize=True,
                     bg_path=None,
                     map_name="nyc"):
        # Set up the components before pomdp loop starts.
        # So that the results of this setup (e.g. the visualizer)
        # can be reused else where.
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
            bg_path = FILEPATHS[map_name]["map_png"]
            viz = mos.MosViz(problem.env, controllable=False, bg_path=bg_path, res=20)
            if viz.on_init() == False:
                raise Exception("Environment failed to initialize")
            viz.update(robot_id,
                       None,
                       None,
                       None,
                       problem.agent.cur_belief)
            viz.on_render()
        return planner, viz


    def _solve(self, problem, planner,
               viz=None,
               step_func=None, # makes this function more extensible.
               step_func_args={},
               max_time=120,  # maximum amount of time allowed to solve the problem
               max_steps=500,
               logging=False): # maximum number of planning steps the agent can take.):
        """
        This function terminates when:
        - maximum time (max_time) reached; This time includes planning and updates
        - agent has planned `max_steps` number of steps
        - agent has taken n FindAction(s) where n = number of target objects.
        """

        # Prepare recording results
        _Rewards = []
        _States = [copy.deepcopy(problem.env.state)]
        # records the expected distance to the true object over the prior, per object
        _ObjDists = {}
        for objid in problem.agent.belief.object_beliefs:
            if objid in problem.env.target_objects:
                belief_obj = problem.agent.belief.object_belief(objid)
                true_obj_pose = problem.env.state.pose(objid)
                expectation = 0
                for obj_state in belief_obj:
                    dist = euclidean_dist(obj_state.pose, true_obj_pose)
                    expectation += dist * belief_obj[obj_state]
                _ObjDists[objid] = expectation


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
            mos.belief_update(problem.agent, real_action, real_observation,
                              problem.env.state.object_states[robot_id],
                              planner)
            _time_used += time.time() - _start


            # Add reward and record find action count
            _total_reward += reward
            if isinstance(real_action, mos.FindAction):
                _find_actions_count += 1

            # Record other information
            _Rewards.append(reward)
            _States.append(copy.deepcopy(problem.env.state))

            # Info and render
            _step_info = "Step %d:  action: %s   reward: %.3f   cum_reward: %.3f"\
                % (i+1, str(real_action), reward, _total_reward)
            if isinstance(planner, pomdp_py.POUCT):
                _step_info += "   NumSims: %d" % planner.last_num_sims
                _step_info += "   PlanTime: %.5f" % planner.last_planning_time

            if logging:
                self.log_event(Event("Trial %s | %s" % (self.name, _step_info)))
            else:
                print(_step_info)

            if viz is not None:
                # This is used to show the sensing range; Not sampled
                # according to observation model.
                robot_pose = problem.env.state.object_states[robot_id].pose
                viz_observation = mos.MosOOObservation({})
                if isinstance(real_action, mos.LookAction) or isinstance(real_action, mos.FindAction):
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
                if logging:
                    self.log_event(Event("Trial %s | Task Finished!\n\n"))
                break
            if _find_actions_count >= len(problem.env.target_objects):
                if logging:
                    self.log_event(Event("Trial %s | Task Ended. FindAction limit reached.\n\n"))
                break
            if _time_used > max_time:
                if logging:
                    self.log_event(Event("Trial %s | Task Ended. Time limit reached.\n\n"))
                break
        results = [RewardsResult(_Rewards),
                   StatesResult(_States),
                   PriorQualityResult(_ObjDists)]
        return results


    def run(self, logging=False):
        """Run the trial"""
        problem_args = self._config["problem_args"]
        ######### NOTE: ADDITIONAL PARAMS #############
        # We are going to impose a dimension parameter to make the world smaller
        problem_args["dimension"] = (41,41)
        ###############################################

        solver_args = self._config["solver_args"]
        exec_args = self._config["exec_args"]
        worldstr = self._config["world"]
        prior_type = self._config["prior_type"]
        language = self._config["language"]
        map_name = self._config["map_name"]
        # A map from a character to an id (e.g. "G": 12)
        # Expecting the object symbols first letters to match keys in this dictionary.
        obj_id_map = self._config["obj_id_map"]
        prior = self._config["prior"]

        print("Language: \"%s\"" % language)
        print("MAP: %s" % map_name)

        landmark_footprints = {}
        sg_dict = {}
        if "prior_metadata" in self._config and "sg_dict" in self._config["prior_metadata"]:
            sg_dict = self._config["prior_metadata"]["sg_dict"]
            print(sg_dict)

        # Create problem
        problem = mos.MosOOPOMDP("r",
                                 prior=prior,
                                 grid_map=worldstr,
                                 obj_id_map=obj_id_map,
                                 **problem_args)

        # Set up solver
        planner, viz = self._setup_solve(problem,
                                         visualize=exec_args["visualize"],
                                         map_name=map_name,
                                        **solver_args)
        if viz is not None:
            img = viz.gridworld_img
            for landmark_symbol in landmark_footprints:
                footprint = mos.rescale_points(landmark_footprints[landmark_symbol],
                                               mos.dim_worldstr(worldstr),
                                               problem_args["dimension"])
                img = mos.MosViz.highlight_gridcells(
                    img, footprint, viz.resolution, color=(50, 50, 50), alpha=0.8)
                viz.update_img(img)

        # Solve it.
        results = self._solve(problem, planner,
                              viz=viz,
                              max_time=exec_args["max_time"],
                              max_steps=exec_args["max_steps"],
                              logging=logging)
        results.append(LangResult(language, sg_dict))
        return results
