from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import sloop
import sloop_object_search.oopomdp.problem as mos
from sloop_object_search.oopomdp.experiments.plotting import *
from sloop_object_search.oopomdp.experiments.pd_utils import *
from sloop_object_search.oopomdp.env.env import *
from sloop.osm.datasets import FILEPATHS, MapInfoDataset
import numpy as np
import math
import json
import yaml
import pickle
import os
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from matplotlib.font_manager import FontProperties
from scipy import stats
import pandas as pd

class StatesResult(PklResult):
    # If you want to visualize prior in trials, set
    # chosen_seeds to 'any'. Otherwise, set it to []
    chosen_seeds = []
    render_path_on_prior = True
    world_size = 41

    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)

    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def collect(cls, path):
        # Collect prior from config
        trial_path = os.path.dirname(path)
        trial_name = os.path.basename(trial_path)
        global_name, seed, specific_name = trial_name.split("_")
        if cls.chosen_seeds == "any" or seed in cls.chosen_seeds:
            with open(os.path.join(trial_path, "trial.pkl"), "rb") as f:
                config = pickle.load(f).config

        # Collect the pickle file
        with open(path, "rb") as f:
            if cls.chosen_seeds == "any" or seed in cls.chosen_seeds:
                return (pickle.load(f), trial_path, config)#["map_name"], prior, prior_type)
            else:
                return (pickle.load(f), trial_path, None)#, None, None)

    @staticmethod
    def compute_path(states):
        # First, figure out robot id
        robot_id = None
        for objid in states[0].object_states:
            if states[-1].object_states[objid].objclass == "robot":
                robot_id = objid
                break

        robot_poses = []
        find_indices = set()
        current_found_set = set()
        for i, state in enumerate(states):
            x,y = state.pose(robot_id)[:2]
            # need to invert y coordinate (for plotting)
            robot_poses.append((x,StatesResult.world_size-y))
            next_found_set = set(states[i].object_states[robot_id].objects_found)
            if len(next_found_set - current_found_set) >= 1:
                find_indices.add(i)
            current_found_set = next_found_set
        return np.array(robot_poses),\
            find_indices  # indices in the robot pose where Find happened

    @staticmethod
    def get_target_states(state):
        target_states = {}
        for objid in state.object_states:
            object_state = state.object_states[objid]
            if object_state.objclass != "robot":
                if object_state.objclass == "target":
                    target_states[objid] = object_state
                    # need to invert y coordinate (for plotting)
                    x,y = object_state.pose
                    target_states[objid]["pose"] = (x,StatesResult.world_size-y)
        return target_states

    @staticmethod
    def visualize_rule_based_belief(ax, objid, belief_obj, color, map_dims):
        """belief is a map from x,y to float"""
        w, l = map_dims
        last_val = -1
        heatmap = np.zeros((w, l, 4))
        for x,y in reversed(sorted(belief_obj, key=belief_obj.get)):
            if last_val != -1:
                color = lighter(color, 1-belief_obj[(x,y)]/last_val)
            if np.mean(color[:3]) < 0.99:
                heatmap[y,x] = np.array(color)  # transpose happens
                last_val = belief_obj[(x,y)]
            if last_val <= 0:
                break
        # invert heatmap y axis
        heatmap = np.flip(heatmap, 0)
        ax.imshow(heatmap, interpolation='gaussian')
        return heatmap

    @staticmethod
    def visualize_keyword_belief(ax, objid, belief_obj, color, map_dims):
        w, l = map_dims
        heatmap = np.zeros((w,l, 4), dtype=float)
        for x in range(w):
            for y in range(l):
                prob = belief_obj[(x,y)]
                heatmap[y,x] = np.array([*color[:3], prob])  # transpose happens
        heatmap = np.flip(heatmap, 0)
        ax.imshow(heatmap)
        return heatmap

    @staticmethod
    def visualize_path(ax, robot_path, find_indices,
                       start_color=(0.75, 1.0, 0.0, 1.0),
                       end_color=(1.0, 0, 1.0, 1.0),
                       find_color=(1.0, 0.89, 0.01)):
        # Plot robot path
        xvals, yvals, colors = [], [], []
        colors = np.array(linear_rgba_gradient(start_color,
                                               end_color, len(robot_path)))
        xvals, yvals = [], []
        find_xvals, find_yvals = [], []
        for i in range(len(robot_path)):
            x,y = robot_path[i][:2]
            xvals.append(x)
            yvals.append(y)
            if i in find_indices:
                find_xvals.append(x)
                find_yvals.append(y)
        # Plot a highlight
        ax.plot(xvals, yvals, "-", color="white", linewidth=18, alpha=0.5)

        # Plot the starting point marker
        ax.scatter([xvals[0]], [yvals[0]], s=530,
                   marker="o", color="black", alpha=0.5, zorder=4)
        ax.scatter([xvals[0]], [yvals[0]], s=480,
                   marker="o", color=start_color, alpha=0.9, zorder=4)

        # Plot the end point marker
        ax.scatter([xvals[-1]], [yvals[-1]], s=480,
                   marker="X", color="black", alpha=0.5, zorder=4)
        ax.scatter([xvals[-1]], [yvals[-1]], s=430,
                   marker="X", color=end_color, alpha=0.7, zorder=4)

        # Plot find markers
        ax.scatter(find_xvals, find_yvals,
                   s=300, marker="*", color="black", alpha=0.5, zorder=5)
        ax.scatter(find_xvals, find_yvals,
                   s=230, marker="*", color=find_color, alpha=0.9, zorder=5)

        # Plot the path
        for i in range(1, len(robot_path)):
            color = colors[i]
            x1,y1 = robot_path[i-1][:2]
            x2,y2 = robot_path[i][:2]
            ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=12, alpha=0.7)

    @staticmethod
    def visualize_targets(ax, target_states, colors):
        # Plot object locations
        for objid in target_states:
            x, y = target_states[objid].pose
            color = np.array([colors[objid]])/255.0
            ax.scatter([x], [y], s=300, marker="o",
                       c="black", zorder=5, alpha=0.9)
            ax.scatter([x], [y], s=100, edgecolor=color, marker="$\mathrm{T}$",
                       c=color, zorder=5)

    @classmethod
    def visualize_prior(cls, ax, map_name, mapinfo,
                        prior, prior_type, target_states,
                        colors):
        # Plot heatmap
        w, l = StatesResult.world_size, StatesResult.world_size#mapinfo.map_dims(map_name)
        for objid in target_states:
            if objid not in prior:
                continue
            color_float = np.array(colors[objid]) / 255.0
            color = [*color_float, 1.0]
            if prior_type.startswith("keyword"):
                cls.visualize_keyword_belief(ax, objid, prior[objid], color, (w,l))

            elif prior_type.startswith("rule#based") and ("ego" in prior_type):
                cls.visualize_rule_based_belief(ax, objid, prior[objid],
                                                color, (w,l))

    @classmethod
    def _get_robot_id(cls, states):
        for objid in states[0].object_states:
            if states[0].object_states[objid].objclass == "robot":
                return objid

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # The number of objects detected & step number when each is detected
        print("Gathering State Results...")
        myresults = {}
        for specific_name in results:
            all_counts = []
            seeds = []
            for seed in results[specific_name]:
                states, _, _ = results[specific_name][seed]
                robot_id = cls._get_robot_id(states)
                objects_found = len(states[0].object_states[robot_id].objects_found)
                founds = []  # A list of tuple (#objfound, step_index) for every increment of #objfound.
                for i, s in enumerate(states):
                    s_r = s.object_states[robot_id]
                    if len(s_r.objects_found) > objects_found:
                        diff = len(s_r.objects_found) - objects_found
                        for j in range(1,diff+1):
                            founds.append((objects_found+j, i))

                seeds.append(seed)
                all_counts.append(founds)
            myresults[specific_name] = {"all_results": all_counts,
                                        "seeds": seeds}

        # The following is for visualization;
        # We compute the path the robot has taken, overlay it
        # on top of the heatmap computed from the prior, on top of map.
        mapinfo = MapInfoDataset()
        colors = {
            12: (50, 168, 82),  # Toyota (green)
            23: (247, 15, 150),  # Bike (pink)
            34: (191, 17, 52)  # Honda (red)
        }
        for specific_name in results:
            for seed in results[specific_name]:
                if not(cls.chosen_seeds == "any" or seed in cls.chosen_seeds):
                    continue
                print("saving visualization for %s_%s" % (specific_name, seed))
                states, trial_path, config = results[specific_name][seed]
                prior = config["prior"]
                prior_type = config["prior_type"]
                prior_metadata = config["prior_metadata"]
                map_name = config["map_name"]
                hint = config["language"]
                if map_name not in mapinfo.landmarks:
                    mapinfo.load_by_name(map_name)
                robot_path, find_indices = cls.compute_path(states)
                target_states = cls.get_target_states(states[0])
                fig, axes = plt.subplots(1, 2, figsize=(16,8))
                w, l = StatesResult.world_size, StatesResult.world_size #mapinfo.map_dims(map_name)
                map_img = mpimg.imread(FILEPATHS[map_name]["map_png"])

                # First, visualize the prior
                axes[0].imshow(map_img, extent=(0, w-1, 0, l-1))#, origin="lower")
                if type(prior) == dict:
                    cls.visualize_prior(axes[0], map_name, mapinfo,
                                        prior, prior_type, target_states, colors)
                    cls.visualize_targets(axes[0], target_states, colors)
                    axes[0].axis("off")
                    axes[0].set_xlim(0, w-1)
                    axes[0].set_ylim(0, l-1)

                # Then, visualize the path on a different map
                if cls.render_path_on_prior:
                    axidx = 0
                else:
                    axidx = 1
                    axes[1].imshow(map_img, extent=(0, w-1, 0, l-1))

                cls.visualize_path(axes[axidx], robot_path, find_indices,
                                      start_color=(0.7, 0.7, 0.7, 1.0),
                                      end_color=(0.1, 0.1, 0.1, 1.0))
                cls.visualize_targets(axes[axidx], target_states, colors)
                axes[axidx].axis("off")
                axes[axidx].set_xlim(0, w-1)
                axes[axidx].set_ylim(0, l-1)

                fig.tight_layout()
                plt.savefig(os.path.join(trial_path,
                                         "path_viz-%s_%s.png"
                                         % (specific_name, seed)), dpi=300)
                plt.clf()

        return myresults

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        # Obtain a table, for each case, the number of
        # detections vs step limit (i.e. success rate)
        df = cls.organize(gathered_results)
        df.to_csv(os.path.join(path, "detections.csv"))

        df_rate, df_rate_map = cls.compute_success_rate(df)
        df_rate.to_csv(os.path.join(path, "detections_success.csv"))
        df_rate_map.to_csv(os.path.join(path, "detections_success_map.csv"))

        df_success_count_steps, df_success_count_steps_map =\
            cls.compute_success_count_by_steps(df)
        df_success_count_steps.to_csv(
            os.path.join(path, "detections_success_steps.csv"))
        df_success_count_steps_map.to_csv(
            os.path.join(path, "detections_success_steps_map.csv"))

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"uniform"},
                                             suffix="step1")

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"informed#5",
                                                        "uniform"},
                                             suffix="step2")

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"informed#5",
                                                        "keyword#auto",
                                                        "uniform"},
                                             suffix="step25")

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"informed#5",
                                                        "keyword",
                                                        "keyword#auto",
                                                        "uniform"},
                                             suffix="step3")

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"informed#5",
                                                        "keyword",
                                                        "keyword#auto",
                                                        "uniform",
                                                        "mixture#full#auto"},
                                             suffix="step35")

        cls.plot_success_count_vs_step_limit(df_success_count_steps,
                                             path, has_map=False,
                                             baselines={"informed#5",
                                                        "keyword",
                                                        "keyword#auto",
                                                        "uniform",
                                                        "mixture#full",
                                                        "mixture#full#auto"},
                                             suffix="step4")

                                                        # "keyword",
                                                        # "keyword#auto",
                                                        # "rule#based#ego#ctx",
                                                        # "rule#based#ego#ctx#auto",
                                                        # "mixture#basic",
                                                        # "mixture#basic#auto",
                                                        # "mixture#full",
                                                        # "mixture#full#auto"})

        # cls.plot_success_count_vs_step_limit(df_success_count_steps,
        #                                      path, has_map=False,
        #                                      baselines={"informed#5",
        #                                                 "uniform",
        #                                                 "keyword",
        #                                                 "keyword#auto",
        #                                                 "rule#based#ego#ctx",
        #                                                 "rule#based#ego#ctx#auto",
        #                                                 "mixture#basic",
        #                                                 "mixture#basic#auto",
        #                                                 "mixture#full",
        #                                                 "mixture#full#auto"})

        # cls.plot_success_count_vs_step_limit(df_success_count_steps_map,
        #                                      path, has_map=True,
        #                                      baselines={"informed#5",
        #                                                 "uniform",
        #                                                 "keyword",
        #                                                 "keyword#auto",
        #                                                 "rule#based#ego#ctx",
        #                                                 "rule#based#ego#ctx#auto",
        #                                                 "mixture#basic",
        #                                                 "mixture#basic#auto",
        #                                                 "mixture#full",
        #                                                 "mixture#full#auto"})


    @classmethod
    def organize(cls, gathered_results):
        """Each case is a (map_name, seed, range). Return a table in pandas dataframe
        with columns:
            sensor_range, prior_type, map_name, seed, founds(list)
        """
        prior_types = set()
        casemap = {}  # map from case to prior_types; Used to make sure we only
                      # use trials where all baselines have results
        for global_name in gathered_results:
            map_name = global_name.split("-")[1]
            results = gathered_results[global_name]
            for specific_name in results:
                prior_type, _ = get_prior_type(specific_name)
                sensor_range = int(specific_name.split("-")[2].split(":")[3][-1])
                prior_types.add(prior_type)

                seeds = results[specific_name]['seeds']
                all_founds = results[specific_name]['all_results']

                for i, seed in enumerate(seeds):
                    # founds is a list of (num_found, step) tuples; We only
                    # care about the step number, because the num_found increases by 1 always.
                    founds = [f[1] for f in all_founds[i]]
                    case = (sensor_range, map_name, seed)
                    if case not in casemap:
                        casemap[case] = []
                    casemap[case].append((prior_type, founds))

        # Make sure we are comparing between cases where all prior types have result.
        rows = []
        counts = {}  # maps from (sensor_range, map_name, prior_type) -> number
        for case in casemap:
            if set(t[0] for t in casemap[case]) != prior_types:
                # We do not have all prior types for this case.
                continue
            for prior_type, founds in casemap[case]:
                sensor_range, map_name, seed = case
                if (sensor_range, map_name, prior_type) not in counts:
                    counts[(sensor_range, map_name, prior_type)] = 0
                rows.append([sensor_range, prior_type, map_name, seed, founds])
                counts[(sensor_range, map_name, prior_type)] += 1
        df = pd.DataFrame(rows, columns=["sensor_range", "prior_type",
                                         "map_name", "seed", "founds"])
        return df


    @classmethod
    def compute_success_rate(cls, df):
        # WARNING: ONLY MEANT TO WORK WHEN THERE IS ONLY ONE TARGET OBJECT
        """Create a table with success rates.

        sensor_range, prior_type, map_name, success_rate
        """
        # First, create a dataframe where the 'founds'
        # list column is replaced by a column indicating
        # number of objects found.
        df = df.copy()
        df['num_found'] = [len(founds) for founds in df['founds']]
        df.drop(columns=["founds"])

        # Next, aggregate by sensor_range, prior_Type,
        # map_name, and take the average on num_found
        agg_map = df.groupby(["sensor_range",
                              "prior_type", "map_name"]).agg(["mean", "std", "count"])
        agg = df.groupby(["sensor_range",
                          "prior_type"]).agg(["mean", "std", "count"])
        return agg, agg_map


    @classmethod
    def compute_success_count_by_steps(cls, df):
        # WARNING: ONLY MEANT TO WORK WHEN THERE IS ONLY ONE TARGET OBJECT
        """Create a table with success rates.

        sensor_range, prior_type, map_name, success_rate
        """
        def get_steps(founds):
            if len(founds) == 1:
                return founds[0]
            else:
                return -1
        # Create two columns: One for the number of steps.
        # One for the number of objects found (1 or 0).
        df = df.copy()
        df['steps'] = [get_steps(founds)
                       for founds in df['founds']]
        df['num_found'] = [len(founds)
                           for founds in df['founds']]
        df.drop(columns=["founds"])

        # Next, aggregate by sensor_range, prior_type,
        # map_name, steps, and take the average on num_found
        agg_map = df.groupby(["sensor_range", "prior_type",
                              "map_name", "steps"]).agg(["mean", "std", "count"])
        agg = df.groupby(["sensor_range", "prior_type",
                          "steps"]).agg(["mean", "std", "count"])

        # flatten the index (i.e. (num_found, mean) becomes "num_found-mean"
        flatten_column_names(agg_map)
        flatten_column_names(agg)

        cls._add_cumulative_count_column(agg)
        cls._add_cumulative_count_column(agg_map, has_map=True)
        return agg, agg_map

    @classmethod
    def _add_cumulative_count_column(cls, df, has_map=False):
        cumcount_column = []  # what goes into the column
        cumcount = 0  # cumulative count
        case = None  # The case we are accumulating for
        for i, row in df.iterrows():
            if row["steps"] < 0:
                cumcount_column.append(-1)
                continue
            if has_map:
                current_case = (row["sensor_range"], row["prior_type"], row["map_name"])
            else:
                current_case = (row["sensor_range"], row["prior_type"])

            if case is None:
                case = current_case
            else:
                if case != current_case:
                    # We encountered a new case
                    case = current_case
                    cumcount = 0
            cumcount += row["num_found-count"]
            cumcount_column.append(cumcount)
        df["cumulative_count"] = cumcount_column

    @classmethod
    def plot_success_count_vs_step_limit(cls, df, path, has_map=False, baselines=None, suffix=""):

        sensor_ranges = df["sensor_range"].unique()
        map_names = ["test"]
        if has_map:
            map_names = df["map_name"].unique()

        label_to_priortype = {}
        for map_name in map_names:
            for sensor_range in sensor_ranges:
                if has_map:
                    df_select = df.loc[(df["sensor_range"] == sensor_range)
                                       & (df["map_name"] == map_name)]
                else:
                    df_select = df.loc[df["sensor_range"] == sensor_range]

                steps = {}  # maps from prior_type to list [steps]
                rates = {}  # maps from prior type to list [success rate]
                for i, row in df_select.iterrows():
                    # Determine the total number of trials ran for
                    # this case, i.e. (sensor_range, prior_type, map_name)
                    prior_type = row["prior_type"]
                    if prior_type not in rates:
                        steps[prior_type] = []
                        rates[prior_type] = []

                    if row["steps"] < 0:
                        continue
                    steps[prior_type].append(row["steps"])
                    rates[prior_type].append(row["cumulative_count"])

                # Plot them
                print("Plotting sucess vs. limit for %d, %s" % (sensor_range, map_name))
                fig = plt.figure(figsize=(3,3))
                for prior_type in steps:
                    if baselines is not None and prior_type not in baselines:
                        continue
                    xvals = steps[prior_type]
                    yvals = rates[prior_type]
                    # To make the plot complete
                    xvals.append(200)
                    yvals.append(yvals[-1])

                    plt.plot(xvals, yvals, color=get_color(prior_type), linestyle=get_line_style(prior_type),
                             zorder=get_zorder(prior_type), label=prior_to_label[prior_type],
                             linewidth=2)
                    label_to_priortype[prior_to_label[prior_type]] = prior_type
                if has_map:
                    title = "sensor range = %d, map_name = %s"\
                        % (sensor_range, map_name)
                else:
                    title = "sensor range = %d"\
                        % sensor_range

                # SORT THE LEGEND
                ax = plt.gca()
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles),
                                              key=lambda t: get_order(label_to_priortype[t[0]])))
                # ax.legend(handles, labels, loc="lower right", prop={'size': 11})

                # plt.ylabel("Number of Completed Search Tasks")
                # plt.xlabel("Number of Steps Allowed Per Search Task")
                plt.title(title)
                plt.xlim(0, 200)
                if has_map is not True:
                    plt.ylim(0, 205)
                plt.tight_layout()
                figname = "detection_success_vs_step_limit-%d-%s%s" % (sensor_range, map_name, suffix)
                plt.savefig(os.path.join(path, figname), dpi=300)
                plt.clf()
