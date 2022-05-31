from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import sloop.oopomdp.problem as mos
from sloop.oopomdp.env.env import *
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
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

# These functions define what show up in tables and plots
def get_prior_type(specific_name):
    if "rule-based" in specific_name:
        prior_type = "rule-based"
        rest = specific_name.split("rule-based-")[1]
    elif "rule#based" in specific_name:
        prior_type = specific_name.split("-")[0]
        rest = "-".join(specific_name.split("-")[1:])
        if "true>blind" in specific_name:
            prior_type = "rule#based#blind"
        if "true>not>blind" in specific_name:
            prior_type = "rule#based#not#blind"
        elif "ego>ctx" in specific_name:
            prior_type = "rule#based#ego#ctx"
        if "#auto" in specific_name:
            prior_type += "#auto"
    elif "informed#" in specific_name:
        prior_type = specific_name.split("-")[0]
        rest = "-".join(specific_name.split("-")[1:])
    else:
        prior_type = specific_name.split("-")[0]
        rest = "-".join(specific_name.split("-")[1:])

    return prior_type, rest

prior_to_label = {
    "informed": "informed(nf)",
    "informed#5": "informed",
    "informed#15": "informed#15",

    "uniform": "uniform",

    "keyword": "keyword, annotated",
    "keyword#auto": "keyword",

    "rule#based#blind": "slu-blind",
    "rule#based#not#blind": "slu-annotated",

    "rule#based#ego#ctx": "slu, annotated",
    "rule#based#ego#ctx#auto": "slu",

    "mixture#basic": "slu-mx1, annotated",
    "mixture#basic#auto": "slu-mx1",

    "mixture#full": "slu-mx2, annotated",
    "mixture#full#auto": "slu-mx2",
}
label_to_prior = {prior_to_label[p]:p for p in prior_to_label}

COLORS = {
    "informed#5":                  "skyblue",
    "informed":                    "teal",
    "informed#15":                 "turquoise",

    "rule#based#ego#ctx":          "gold",
    "rule#based#ego#ctx#auto":     "gold",

    "rule#based#not#blind":        "plum",
    "rule#based#blind":            "lightcoral",

    "rule#based":                  "orange",

    "keyword":                     "green",
    "keyword#auto":                "green",

    "uniform":                     "silver",

    "mixture#basic":               "red",
    "mixture#basic#auto":          "red",

    "mixture#full":                "#800000",
    "mixture#full#auto":           "#800000",
}

def get_lighter_color(prior_type):
    # TODO: Actually not sure what this is used for
    if prior_type in COLORS:
        return COLORS[prior_type]
    else:
        for x in COLORS:
            if prior_type.startswith(x):
                return COLORS[x]

def get_color(prior_type):
    if prior_type in COLORS:
        return COLORS[prior_type]
    else:
        for x in COLORS:
            if prior_type.startswith(x):
                return COLORS[x]

def get_order(prior_type):
    """Order of the barchart arrangement"""
    ORDERS = {
        "informed":                0,
        "informed#5":              1,
        "informed#15":             2,
        "mixture#full":            4,
        "mixture#full#auto":       3,
        "mixture#basic":           6,
        "mixture#basic#auto":      5,
        "rule#based#not#blind":    7,
        "rule#based#blind":        8,
        "rule#based#ego#ctx":      10,
        "rule#based#ego#ctx#auto": 9,
        "keyword":                 12,
        "keyword#auto":            11,
        "uniform":                 13
    }
    if prior_type in ORDERS:
        return ORDERS[prior_type]
    else:
        for x in ORDERS:
            if prior_type.startswith(x):
                return ORDERS[x]

def get_zorder(prior_type):
    ORDERS = {
        "informed":                1,
        "informed#5":              1,
        "informed#15":             1,
        "rule#based#not#blind":    1,
        "rule#based#blind":        2,
        "rule#based#ego#ctx":      4,
        "rule#based#ego#ctx#auto": 4,
        "mixture#full":            6,
        "mixture#full#auto":       6,
        "mixture#basic":           5,
        "mixture#basic#auto":      5,
        "keyword":                 3,
        "keyword#auto":            3,
        "uniform":                 0
    }
    if prior_type in ORDERS:
        return ORDERS[prior_type]
    else:
        for x in ORDERS:
            if prior_type.startswith(x):
                return ORDERS[x]

def get_line_style(prior_type):
    """Order of the barchart arrangement"""
    ORDERS = {
        "informed":                "dashed",
        "informed#5":              "solid",
        "informed#15":             "dashdot",
        "mixture#full":            "dashed",
        "mixture#full#auto":       "solid",
        "mixture#basic":           "dashed",
        "mixture#basic#auto":      "solid",
        "rule#based#not#blind":    "dotted",
        "rule#based#blind":        "dotted",
        "rule#based#ego#ctx":      "dashed",
        "rule#based#ego#ctx#auto": "solid",
        "keyword":                 "dashed",
        "keyword#auto":            "solid",
        "uniform":                 "solid"
    }
    if prior_type in ORDERS:
        return ORDERS[prior_type]
    else:
        for x in ORDERS:
            if prior_type.startswith(x):
                return ORDERS[x]

def ci_normal(series, confidence_interval=None, c=0.95):
    ### CODE BY CLEMENT at LIS ###
    """Confidence interval for normal distribution with
    unknown mean and variance. `confidence_interval` and `c`
    are the same parameter.
    """
    if confidence_interval is None:
        confidence_interval = c

    series = np.asarray(series)
    # this is the "percentage point function" which is the inverse of a cdf
    # divide by 2 because we are making a two-tailed claim
    tscore = stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)
    y_error = stats.sem(series)
    half_width = y_error * tscore
    return half_width

def stderr(series):
    """computes the standard error of the mean"""
    return stats.sem(series)

ALL_MAPS = {"cleveland", "denver", "austin", "honolulu", "washington_dc"}


####### MEAN CI PLOT #########
def do_subplot_mean_ci(map_name, means, errs, xvals, ax, width=0.2, plot_type="rewards"):
    objects = []
    labels = []
    for i, prior_type in enumerate(sorted(means, key=lambda x:get_order(x))):
        y_meth = [means[prior_type][s] for s in xvals]
        yerr_meth = [errs[prior_type][s] for s in xvals]
        label = prior_to_label[prior_type]
        obj = ax.bar(xvals + width*i, y_meth, width, yerr=yerr_meth, label=label,
                     color=get_color(prior_type))

        ax.set_title(map_name)
        # ax.set_xlabel("Sensor Range")
        ax.set_xticks(xvals + width*(len(means)-1)/2)
        ax.set_xticklabels(xvals)
        if plot_type == "rewards":
            ax.set_ylim(-500, 2000)
            ax.set_ylabel("Cumulative Reward")
        elif plot_type == "detections":
            ax.set_ylabel("Number of Detected Objects")
        objects.append(obj)
        labels.append(prior_type)
    return objects, labels

def plot_langprior(gathered_results,
                   suffix="plot",
                   plot_type="rewards",
                   train_maps={}, test_maps={},
                   subplot_width=6,
                   subplot_length=6,
                   bar_width=0.2,
                   skip_city_maps=False,
                   ncols=4):

    # Create a Nx4 plot; Each plot is for a specific map.
    # and finally, plot the overall result for train_maps and test_maps.
    all_maps = set()
    for global_name in gathered_results:
        chunks = global_name.split("-")
        if len(chunks) == 2:
            map_name = chunks[1]
            all_maps.add(map_name)
        else:
            all_maps.add(global_name)

    nplots = 0
    maps_plotting = []
    if not skip_city_maps:
        nplots = len(all_maps)
        maps_plotting = list(sorted(all_maps))
    if len(train_maps) > 0:
        nplots += 1
        maps_plotting.append("train")
    if len(test_maps) > 0:
        nplots += 1
        maps_plotting.append("test")
    nrows = int(math.ceil( nplots / ncols))
    fig, axeslist = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(subplot_width*ncols, subplot_length*nrows),
                                 sharey=True)

    axes = {}
    for i, map_name in enumerate(maps_plotting):
        rowidx = i // ncols
        colidx = i - rowidx * ncols
        if nrows > 1:
            if ncols > 1:
                axes[map_name] = axeslist[rowidx][colidx]
            else:
                axes[map_name] = axeslist[rowidx]
        else:
            if ncols > 1:
                axes[map_name] = axeslist[colidx]
            else:
                axes[map_name] = axeslist


    train_trial_results = {}  # map from method_name -> [trial results]
    test_trial_results = {}  # map from method_name -> [trial results]
    for global_name in gathered_results:
        chunks = global_name.split("-")
        xvals = []
        means = {}
        errs = {}
        if len(chunks) == 2:
            map_name = chunks[1]
        else:
            map_name = global_name

        # Gather results per method
        results = gathered_results[global_name]
        for specific_name in results:
            prior_type, rest = get_prior_type(specific_name)
            sensor_range = int(rest.split("-")[1].split(":")[3][-1])
            if sensor_range not in xvals:
                xvals.append(sensor_range)

            if prior_type not in means:
                means[prior_type] = {}
                errs[prior_type] = {}
            means[prior_type][sensor_range] = results[specific_name]["mean"]
            errs[prior_type][sensor_range] = results[specific_name]["conf-95"]

            # We want to also compute mean/err over all training/test trials,
            # not over training/test maps (because there are too few maps).
            if map_name in train_maps:
                if prior_type not in train_trial_results:
                    train_trial_results[prior_type] = {}
                if sensor_range not in train_trial_results[prior_type]:
                    train_trial_results[prior_type][sensor_range] = []
                train_trial_results[prior_type][sensor_range].extend(results[specific_name]["all_results"])
            if map_name in test_maps:
                if prior_type not in test_trial_results:
                    test_trial_results[prior_type] = {}
                if sensor_range not in test_trial_results[prior_type]:
                    test_trial_results[prior_type][sensor_range] = []
                test_trial_results[prior_type][sensor_range].extend(results[specific_name]["all_results"])

        # Also used later (plotting train/test)
        xvals = np.array(sorted(xvals))
        if not skip_city_maps:
            ax = axes[map_name]
            ax_objs, ax_labels = do_subplot_mean_ci(map_name, means, errs, xvals, ax, width=bar_width, plot_type=plot_type)

    # Plot by train/test maps
    train_means = {}
    train_errs = {}
    test_means = {}
    test_errs = {}
    for prior_type in train_trial_results:
        train_means[prior_type] = {}
        train_errs[prior_type] = {}
        for sensor_range in train_trial_results[prior_type]:
            ci95 = ci_normal(train_trial_results[prior_type][sensor_range], confidence_interval=0.95)
            train_means[prior_type][sensor_range] = np.mean(train_trial_results[prior_type][sensor_range])
            train_errs[prior_type][sensor_range] = ci95
    for prior_type in test_trial_results:
        test_means[prior_type] = {}
        test_errs[prior_type] = {}
        for sensor_range in test_trial_results[prior_type]:
            ci95 = ci_normal(test_trial_results[prior_type][sensor_range], confidence_interval=0.95)
            test_means[prior_type][sensor_range] = np.mean(test_trial_results[prior_type][sensor_range])
            test_errs[prior_type][sensor_range] = ci95
    if "train" in maps_plotting:
        ax_objs, ax_labels = do_subplot_mean_ci("train", train_means, train_errs, xvals, axes["train"], plot_type=plot_type,
                                        width=bar_width)
    if "test" in maps_plotting:
        ax_objs, ax_labels = do_subplot_mean_ci("test", test_means, test_errs, xvals, axes["test"], plot_type=plot_type,
                                        width=bar_width)

    # Get the legend
    if hasattr(axeslist, "__len__"):
        lgd = fig.legend(tuple(ax_objs), tuple(ax_labels), loc='center right')
    else:
        # Shrink current axis by 20%
        fontP = FontProperties()
        fontP.set_size("small")
        ax = axeslist
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.75])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.53), prop=fontP)

    if plot_type == "rewards":
        fig.savefig("rewards-%s.png" % suffix, dpi=300)
    elif plot_type == "detections":
        fig.savefig("detections-%s.png" % suffix, dpi=300)


############ BOX PLOT #############
def do_subplot_boxplot(map_name, all_results, xvals, ax, width=0.2, plot_type="rewards"):
    objects = []
    labels = []

    boxprops = dict(linestyle='-', linewidth=0.0, color="dimgray")
    meanlineprops = dict(linestyle='--', linewidth=1.0, color="dimgray")
    for i, prior_type in enumerate(sorted(all_results, key=lambda x:get_order(x))):
        # Will create n boxplots, one for each sensor range (i.e. value in xvals)

        color = get_color(prior_type)
        lighter_color = get_lighter_color(prior_type)
        medianprops = dict(linestyle='-', linewidth=2.0, color=lighter_color)
        for s in xvals:
            data = all_results[prior_type][s]
            obj = ax.boxplot(data,
                             # showmeans=True, meanline=True,
                             positions=[s+(width)*i],
                             medianprops=medianprops,
                             meanprops=meanlineprops,
                             boxprops=boxprops,
                             showfliers=False,
                             patch_artist=True,
                             widths=width)
            for patch in obj["boxes"]:
                patch.set_facecolor(color)
            ## change color and linewidth of the caps
            for cap in obj['caps']:
                cap.set(color="black", linewidth=0.2)
            ## change color and linewidth of the whiskers
            for whisker in obj['whiskers']:
                whisker.set(color='black', linewidth=0.2)

        ax.set_title(map_name)
        # ax.set_xlabel("Sensor Range")
        ax.set_xticks(xvals + (width)*(len(all_results)-1)/2)
        ax.set_xticklabels(xvals)
        if plot_type == "rewards":
            # ax.set_ylim(-500, 2100)
            ax.set_ylabel("Cumulative Reward")
        elif plot_type == "detections":
            ax.set_ylabel("Number of Detected Objects")
        objects.append(obj)
        labels.append(prior_type)
    return objects, labels


def plot_langprior_boxplot(gathered_results,
                           suffix="plot",
                           plot_type="rewards",
                           train_maps={}, test_maps={},
                           subplot_width=6,
                           subplot_length=6,
                           bar_width=0.2,
                           skip_city_maps=False,
                           ncols=4):

    # Create a Nx4 plot; Each plot is for a specific map.
    # and finally, plot the overall result for train_maps and test_maps.
    all_maps = set()
    for global_name in gathered_results:
        chunks = global_name.split("-")
        if len(chunks) == 2:
            map_name = chunks[1]
            all_maps.add(map_name)
        else:
            all_maps.add(global_name)

    nplots = 0
    maps_plotting = []
    if not skip_city_maps:
        nplots = len(all_maps)
        maps_plotting = list(sorted(all_maps))
    if len(train_maps) > 0:
        nplots += 1
        maps_plotting.append("train")
    if len(test_maps) > 0:
        nplots += 1
        maps_plotting.append("test")
    nrows = int(math.ceil( nplots / ncols))
    fig, axeslist = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(subplot_width*ncols, subplot_length*nrows),
                                 sharey=True)

    axes = {}
    for i, map_name in enumerate(maps_plotting):
        rowidx = i // ncols
        colidx = i - rowidx * ncols
        if nrows > 1:
            if ncols > 1:
                axes[map_name] = axeslist[rowidx][colidx]
            else:
                axes[map_name] = axeslist[rowidx]
        else:
            if ncols > 1:
                axes[map_name] = axeslist[colidx]
            else:
                axes[map_name] = axeslist


    train_trial_results = {}  # map from method_name -> [trial results]
    test_trial_results = {}  # map from method_name -> [trial results]
    for global_name in gathered_results:
        chunks = global_name.split("-")
        xvals = []
        all_results = {} # maps from method_name -> {sensor_range -> all_results}
        if len(chunks) == 2:
            map_name = chunks[1]
        else:
            map_name = global_name

        # Gather results per method
        results = gathered_results[global_name]
        for specific_name in results:
            prior_type, rest = get_prior_type(specific_name)
            sensor_range = int(rest.split("-")[1].split(":")[3][-1])
            if sensor_range not in xvals:
                xvals.append(sensor_range)

            if prior_type not in all_results:
                all_results[prior_type] = {}
            all_results[prior_type][sensor_range] = results[specific_name]["all_results"]

            # We want to also compute mean/err over all training/test trials,
            # not over training/test maps (because there are too few maps).
            if map_name in train_maps:
                if prior_type not in train_trial_results:
                    train_trial_results[prior_type] = {}
                if sensor_range not in train_trial_results[prior_type]:
                    train_trial_results[prior_type][sensor_range] = []
                train_trial_results[prior_type][sensor_range].extend(results[specific_name]["all_results"])
            if map_name in test_maps:
                if prior_type not in test_trial_results:
                    test_trial_results[prior_type] = {}
                if sensor_range not in test_trial_results[prior_type]:
                    test_trial_results[prior_type][sensor_range] = []
                test_trial_results[prior_type][sensor_range].extend(results[specific_name]["all_results"])

        # Also used later (plotting train/test)
        xvals = np.array(sorted(xvals))
        if not skip_city_maps:
            ax = axes[map_name]
            ax_objs, ax_labels = do_subplot_boxplot(map_name, all_results,
                                                    xvals, ax, width=bar_width,
                                                    plot_type=plot_type)

    # Plot by train/test maps
    if "train" in maps_plotting:
        ax_objs, ax_labels = do_subplot_boxplot("train", train_trial_results,
                                                xvals, axes["train"], plot_type=plot_type,
                                                width=bar_width)
    if "test" in maps_plotting:
        ax_objs, ax_labels = do_subplot_boxplot("test", test_trial_results,
                                                xvals, axes["test"], plot_type=plot_type,
                                                width=bar_width)

    # Get the legend
    if hasattr(axeslist, "__len__"):
        lgd = fig.legend(tuple(ax_objs), tuple(ax_labels), loc='center right')
    else:
        # Shrink current axis by 20%
        fontP = FontProperties()
        fontP.set_size("small")
        ax = axeslist
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.75])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.53), prop=fontP)

    if plot_type == "rewards":
        fig.savefig("rewards-boxplot-%s.png" % suffix, dpi=300)
    elif plot_type == "detections":
        fig.savefig("detections-boxplot-%s.png" % suffix, dpi=300)



# Plot results for every case, together on a line plot
def plot_case_results(case_results, suffix="plot",
                      subplot_width=40, subplot_length=4):
    sensor_ranges = set()
    for case in case_results:
        map_name, seed, sensor_range = case.split("-")
        sensor_ranges.add(sensor_range)

    prior_range_results = {}  # Map from (prior_type, range) -> [reward ...]
    fig, axeslist = plt.subplots(nrows=len(sensor_ranges), ncols=1,
                             figsize=(subplot_width, subplot_length*len(sensor_ranges)))
    axes = {}
    for i, r in enumerate(sorted(sensor_ranges)):
        if hasattr(axeslist, "__len__"):
            axes[r] = axeslist[i]
        else:
            # When there's only one sensor_range
            axes[r] = axeslist

    for i, case in enumerate(case_results):
        map_name, seed, sensor_range = case.split("-")
        for prior_type in case_results[case]:
            cum_reward = case_results[case][prior_type]
            if (prior_type, sensor_range) not in prior_range_results:
                prior_range_results[(prior_type, sensor_range)] = []
            prior_range_results[(prior_type, sensor_range)].append(cum_reward)

    for prior_type, sensor_range in prior_range_results:
        rewards = prior_range_results[(prior_type, sensor_range)]
        xvals = np.arange(len(rewards))
        label = prior_to_label[prior_type]
        axes[sensor_range].plot(xvals, rewards, "o-", label=label,
                                color=get_color(prior_type), alpha=0.7,
                                zorder=get_zorder(prior_type))
        axes[sensor_range].set_title("Sensor range %s" % sensor_range)
        axes[sensor_range].legend(loc='lower right')
        # axes[sensor_range].set_ylim(1200, 2000)
    fig.savefig("rewards_by_case-%s.png" % suffix)


################# Colors ##################
def linear_rgb_gradient(rgb_start, rgb_end, n):
    colors = [rgb_start]
    for t in range(1, n):
        colors.append(tuple(
            rgb_start[i] + float(t)/(n-1)*(rgb_end[i] - rgb_start[i])
            for i in range(3)
        ))
    return colors

def linear_rgba_gradient(rgba_start, rgba_end, n):
    colors = [rgba_start]
    for t in range(1, n):
        colors.append(tuple(
            rgba_start[i] + float(t)/(n-1)*(rgba_end[i] - rgba_start[i])
            for i in range(4)
        ))
    return colors


def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([1.0, 1.0, 1.0, 0.0])
    vector = white-color
    return color + vector * percent
