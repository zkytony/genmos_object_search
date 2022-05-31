from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import sloop.oopomdp.problem as mos
from sloop.oopomdp.env.env import *
from .plotting import *
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


class PriorQualityResult(YamlResult):
    def __init__(self, obj_dists):
        """list of state objects"""
        super().__init__(obj_dists)

    @classmethod
    def FILENAME(cls):
        return "prior_quality.yaml"

    @classmethod
    def collect(cls, path):
        with open(path) as f:
            return yaml.load(f, Loader=yaml.Loader)

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # Returns the number of objects detected at the end.
        myresult = {}
        for specific_name in results:
            all_dists = []
            for seed in results[specific_name]:
                expected_obj_dist = results[specific_name][seed]
                for objid in expected_obj_dist:
                    all_dists.append(expected_obj_dist[objid])
            sample_size = len(all_dists)
            ci95 = ci_normal(all_dists, confidence_interval=0.95)
            sem = stderr(all_dists)
            myresult[specific_name] = {'mean': np.mean(all_dists),
                                       'std': np.std(all_dists),
                                       'conf-95': ci95,
                                       'sem': sem,
                                       '_size': len(all_dists)}
        return myresult

    @classmethod
    def save_gathered_results(cls, gathered_results, path):

        # Save latex table for this
        def _tex_tab_val(entry, bold=False):
            pm = "$\pm$" if not bold else "$\\bm{\pm}$"
            return "%.2f %s %.2f" % (entry["mean"], pm, entry["std"])

        with open(os.path.join(path, "prior_quality.json"), "w") as f:
            json.dump(gathered_results, f, indent=4, sort_keys=True)

        table = {}
        for global_name in gathered_results:
            for specific_name in gathered_results[global_name]:
                result = gathered_results[global_name][specific_name]
                prior_type, _ = get_prior_type(specific_name)
                mean, ci, sem = result["mean"], result["conf-95"], result["sem"]
                if prior_type in table:
                    if table[prior_type]["mean"] <= mean:
                        continue
                table[prior_type] = {"mean":mean, "ci": ci, "sem": sem}
        df = pd.DataFrame(table)
        df.to_csv(os.path.join(path, "prior_quality.csv"))
