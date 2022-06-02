from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import sloop.oopomdp.problem as mos
from sloop.oopomdp.env.env import *
from .plotting import *
from .pd_utils import *
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


class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        super().__init__(rewards)
    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def discounted_reward(cls, rewards, gamma=0.95):
        discount = 1.0
        cum_disc = 0.0
        for reward in rewards:
            cum_disc += discount * reward
            discount *= gamma
        return cum_disc

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # compute cumulative rewards
        myresult = {}
        for specific_name in sorted(results):
            all_rewards = []
            all_discounted_rewards = []
            all_num_detected = []
            seeds = []
            for seed in sorted(results[specific_name]):
                trial_rewards = list(results[specific_name][seed])
                cum_reward = sum(trial_rewards)
                all_rewards.append(cum_reward)

                disc_reward = cls.discounted_reward(trial_rewards, gamma=0.95)
                all_discounted_rewards.append(disc_reward)

                num_detected = 0
                for reward in trial_rewards:
                    if reward == 1000:
                        num_detected += 1
                all_num_detected.append(num_detected)

                seeds.append(seed)

            sample_size = len(all_rewards)
            myresult[specific_name] = {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'conf-95': ci_normal(all_rewards, confidence_interval=0.95),
                'disc_mean': np.mean(all_discounted_rewards),
                'disc_std': np.std(all_discounted_rewards),
                'disc_conf-95': ci_normal(all_discounted_rewards, confidence_interval=0.95),
                '_size': sample_size,
                'all_results': all_rewards,
                'all_results_discounted': all_discounted_rewards,
                'all_num_detected': all_num_detected,
                'seeds': seeds}
        return myresult


    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])

        # Save plain text
        with open(os.path.join(path, "rewards.json"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)

        # Save result for every case, per baseline
        case_results, case_results_discounted = cls.organize_by_case(gathered_results)
        with open(os.path.join(path, "rewards_by_case.yaml"), "w") as f:
            yaml.dump(case_results, f)
        with open(os.path.join(path, "rewards_by_case_discounted.yaml"), "w") as f:
            yaml.dump(case_results_discounted, f)
        # plot_case_results(case_results)

        counts = cls.case_wise_stats(case_results,
                                     our_method=("rule#based#ego#ctx", "slu"),
                                     other_method=("keyword", "keyword"))
        with open(os.path.join(path, "rewards_compare_counts-slu_keyword.yaml"), "w") as f:
            yaml.dump(counts, f)
        counts = cls.case_wise_stats(case_results_discounted,
                                     our_method=("rule#based#ego#ctx", "slu"),
                                     other_method=("keyword", "keyword")        )
        with open(os.path.join(path, "rewards_compare_counts_discounted-slu_keyword.yaml"), "w") as f:
            yaml.dump(counts, f)


        counts = cls.case_wise_stats(case_results,
                                     our_method=("rule#based#ego#ctx#auto", "slu#auto"),
                                     other_method=("keyword#auto", "keyword#auto"))
        with open(os.path.join(path, "rewards_compare_counts-slu_keyword#auto.yaml"), "w") as f:
            yaml.dump(counts, f)
        counts = cls.case_wise_stats(case_results_discounted,
                                     our_method=("rule#based#ego#ctx#auto", "slu#auto"),
                                     other_method=("keyword#auto", "keyword#auto")        )
        with open(os.path.join(path, "rewards_compare_counts_discounted-slu_keyword#auto.yaml"), "w") as f:
            yaml.dump(counts, f)


        counts = cls.case_wise_stats(case_results,
                                     our_method=("informed", "informed"),
                                     other_method=("informed#5", "informed#5"))
        with open(os.path.join(path, "rewards_compare_counts-inf_inf#5.yaml"), "w") as f:
            yaml.dump(counts, f)
        counts = cls.case_wise_stats(case_results_discounted,
                                     our_method=("informed", "informed"),
                                     other_method=("informed#5", "informed#5"))
        with open(os.path.join(path, "rewards_compare_counts_discounted-inf_inf#5.yaml"), "w") as f:
            yaml.dump(counts, f)

        # if os.path.basename(path).lower().startswith("foref")\
        #    or os.path.basename(path).lower().startswith("eval"):
        #     test_map = os.path.basename(path).lower().split("_")[0].split("-")[1]
        #     plot_langprior(gathered_results, suffix="plot", plot_type="rewards",
        #                    train_maps={}, test_maps={test_map})

        # elif os.path.basename(path).lower().startswith("joint"):
        #     plot_langprior(gathered_results, suffix="joint-plot", plot_type="rewards",
        #                    train_maps={}, test_maps=ALL_MAPS, skip_city_maps=True, ncols=1,
        #                    subplot_width=6, subplot_length=6, bar_width=0.15)
        #     plot_langprior(gathered_results, suffix="citywise-plot", plot_type="rewards",
        #                    train_maps={}, test_maps={}, ncols=5, bar_width=0.15)

        #     plot_langprior_boxplot(gathered_results, suffix="joint-plot", plot_type="rewards",
        #                            train_maps={}, test_maps=ALL_MAPS, skip_city_maps=True, ncols=1,
        #                            subplot_width=6, subplot_length=6, bar_width=0.12)
        #     plot_langprior_boxplot(gathered_results, suffix="citywise-plot", plot_type="rewards",
        #                            train_maps={}, test_maps={}, ncols=5, bar_width=0.12)

        df = cls.organize(gathered_results)
        df_reward, df_reward_map = cls.compute_reward_summary(df, num_detected=1)
        df_reward.to_csv(os.path.join(path, "reward_summary.csv"))
        df_reward_map.to_csv(os.path.join(path, "reward_summary_map.csv"))


    ######### It's sad that these code are not using Pandas #####
    @classmethod
    def organize_by_case(cls, gathered_results):
        """Each case is a (map_name, seed, range)"""
        case_results = {}
        case_results_discounted = {}
        for global_name in gathered_results:
            map_name = global_name.split("-")[1]
            results = gathered_results[global_name]
            for specific_name in results:
                prior_type, _ = get_prior_type(specific_name)
                sensor_range = int(specific_name.split("-")[2].split(":")[3][-1])

                seeds = results[specific_name]['seeds']
                all_rewards = results[specific_name]['all_results']
                all_discounted_rewards = results[specific_name]['all_results_discounted']

                for i, seed in enumerate(seeds):
                    case = "%s-%s-%s" % (map_name, seed, sensor_range)
                    cum_reward = all_rewards[i]
                    cum_reward_discounted = all_discounted_rewards[i]
                    if case not in case_results:
                        case_results[case] = {}
                        case_results_discounted[case] = {}
                    case_results[case][prior_type] = cum_reward
                    case_results_discounted[case][prior_type] = cum_reward_discounted
        return case_results, case_results_discounted

    @classmethod
    def case_wise_stats(cls, case_results,
                        our_method=("rule#based#ego#ctx", "slu"),
                        other_method=("keyword", "keyword")):
        counts = {'_better_': 0,
                  '_worse_': 0,
                  '_even_': 0,
                  '_DETAILS_': {'_better_': {},
                              '_worse_': {},
                              '_even_': {}}}
        diffs_better = []
        diffs_worse = []
        diffs = []   # overall
        diffs3 = []  # for sensor range 3
        diffs4 = []  # for sensor range 4
        diffs5 = []  # for sensor range 5
        our_method, our_abrv = our_method
        other_method, other_abrv = other_method
        for case in case_results:
            if our_method in case_results[case]\
               and other_method in case_results[case]:
                map_name = case.split("-")[0]
                ours = case_results[case][our_method]
                other = case_results[case][other_method]
                if map_name not in counts:
                    counts[map_name] = {'_better_': 0,
                                        '_worse_': 0,
                                        '_even_': 0}
                if ours > other:
                    counts["_better_"] += 1
                    counts[map_name]["_better_"] += 1
                    counts['_DETAILS_']['_better_'][case] = {our_abrv: ours, other_abrv: other}
                    diffs_better.append(ours - other)
                elif ours < other:
                    counts["_worse_"] += 1
                    counts[map_name]["_worse_"] += 1
                    counts['_DETAILS_']['_worse_'][case] = {our_abrv: ours, other_abrv: other}
                    diffs_worse.append(other - ours)
                else:
                    counts["_even_"] += 1
                    counts[map_name]["_even_"] += 1
                    counts['_DETAILS_']['_even_'][case] = {our_abrv: ours, other_abrv: other}

                diffs.append(ours - other)
                sensor_range = int(case.split("-")[2])
                if sensor_range == 3:
                    diffs3.append(ours - other)
                elif sensor_range == 4:
                    diffs4.append(ours - other)
                elif sensor_range == 5:
                    diffs5.append(ours - other)

        counts["_DIFFS_"] = {"_better_": "%.3f +/- %.3f" % (np.mean(diffs_better), np.std(diffs_better)),
                             "_worse_": "%.3f +/- %.3f" % (np.mean(diffs_worse), np.std(diffs_worse)),
                             "_overall_": "%.3f +/- %.3f (%.3f)" % (np.mean(diffs), np.std(diffs), ci_normal(diffs, c=0.95)),
                             "_range3_": "%.3f +/- %.3f (%.3f)" % (np.mean(diffs3), np.std(diffs3), ci_normal(diffs3, c=0.95)),
                             "_range4_": "%.3f +/- %.3f (%.3f)" % (np.mean(diffs4), np.std(diffs4), ci_normal(diffs4, c=0.95)),
                             "_range5_": "%.3f +/- %.3f (%.3f)" % (np.mean(diffs5), np.std(diffs5), ci_normal(diffs5, c=0.95))}
        return counts

    ####### But these will be using pandas #######
    @classmethod
    def organize(cls, gathered_results):
        """Create Pandas dataframe with columns
        (sensor_range, prior_type, map_name, seed, cum_reward, disc_reward)"""
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
                all_rewards = results[specific_name]['all_results']
                all_discounted_rewards = results[specific_name]['all_results_discounted']
                all_num_detected = results[specific_name]['all_num_detected']

                for i, seed in enumerate(seeds):
                    case = (sensor_range, map_name, seed)
                    cum_reward = all_rewards[i]
                    cum_reward_discounted = all_discounted_rewards[i]
                    num_detected = all_num_detected[i]
                    if case not in casemap:
                        casemap[case] = []
                    casemap[case].append((prior_type, num_detected, cum_reward, cum_reward_discounted))
        # Make sure we are comparing between cases where all prior types have result.
        rows = []
        counts = {}  # maps from (sensor_range, map_name, prior_type) -> number
        for case in casemap:
            if set(t[0] for t in casemap[case]) != prior_types:
                # We do not have all prior types for this case.
                continue
            for prior_type, num_detected, cum_reward, cum_reward_discounted in casemap[case]:
                sensor_range, map_name, seed = case
                if (sensor_range, map_name, prior_type) not in counts:
                    counts[(sensor_range, map_name, prior_type)] = 0
                if counts[(sensor_range, map_name, prior_type)] >= 40:
                    continue  # already 40 counted
                rows.append([sensor_range, prior_type, map_name, seed, num_detected,
                             cum_reward, cum_reward_discounted])
                counts[(sensor_range, map_name, prior_type)] += 1
        df = pd.DataFrame(rows, columns=["sensor_range", "prior_type",
                                         "map_name", "seed", "num_detected", "cum_reward", "disc_reward"])
        return df


    @classmethod
    def compute_reward_summary(cls, df, num_detected=-1):
        """If `num_detected` is greater than 0, then only rewards
        with at least this number of detections will be summarized."""
        def aggregate(grouped):
            agg = grouped.agg([("ci95", lambda x: ci_normal(x, confidence_interval=0.95)),
                               ("ci90", lambda x: ci_normal(x, confidence_interval=0.90)),
                               ("sem", lambda x: stderr(x)),
                               ('avg', 'mean'),
                               "std",
                               "count"])
            flatten_column_names(agg)
            return agg

        df = df.copy()
        if num_detected > 0:
            df = df.loc[df["num_detected"] >= num_detected]

        grouped = df.groupby(["sensor_range", "prior_type"])
        agg = aggregate(grouped)

        grouped_map = df.groupby(["sensor_range", "prior_type", "map_name"])
        agg_map = aggregate(grouped_map)

        return agg, agg_map
