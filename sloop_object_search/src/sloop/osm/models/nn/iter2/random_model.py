import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sloop.utils import json_safe
from sloop.models.nn.base_model import BaseModel
from sloop.models.nn.loss_function import FoRefAngleLoss, clamp_angle
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
from sloop.models.heuristics.rules import ForefRule
from sloop.models.heuristics.model import evaluate as rule_based_evaluate
from sloop.models.heuristics.model import RuleBasedModel
from sloop.models.nn.iter2.common import get_common_data
import numpy as np
import json
from pprint import pprint
import math

class RandomModel(BaseModel):
    NAME="random"

    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(RandomModel, self).__init__()
        self.keyword = keyword
        self.map_dims = map_dims

    def forward(self, x):
        # Note that this is not normalized.
        return math.radians(random.uniform(0, 360))

    @classmethod
    def get_data(cls, keyword, data_dirpath, map_names,
                 **kwargs):
        return get_common_data(keyword, data_dirpath, map_names, **kwargs)

    @classmethod
    def Train(cls, model, trainset, device,
              **kwargs):
        print("NO TRAINING REQUIRED")


    @classmethod
    def Eval(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        pass

    @classmethod
    def Plot(cls, keyword, model, dataset, device,
             save_dirpath, suffix="plot", **kwargs):
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())
        batch_size = 1
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        # Number of examples to generate plots for. The
        # rest would only be used to evaluate the metrics.
        plot_amount = kwargs.get("plot_amount", 10)
        map_dims = kwargs.get("map_dims", (21,21))
        relative_origin = kwargs.get("relative", False)

        plot_indices = list(np.arange(len(dataset)))
        random.Random(100).shuffle(plot_indices)
        plot_indices = set(plot_indices[:plot_amount])

        plots_dir = os.path.join(save_dirpath, "plots", suffix)
        if not os.path.exists(os.path.join(plots_dir)):
            os.makedirs(os.path.join(plots_dir))
        metrics_dir = os.path.join(save_dirpath, "metrics", suffix)
        if not os.path.exists(os.path.join(metrics_dir)):
            os.makedirs(os.path.join(metrics_dir))

        # Saving the metrics
        results = {"true_pred_angle_diff": [],
                   "true_pred_origin_diff": []}

        for i, batch in enumerate(data_loader):
            prediction_angle = model(None)
            foref_pred = np.array([*batch[FdFoRefOrigin.NAME][0].numpy(),
                                   prediction_angle])
            foref_pred = read_foref(dataset, foref_pred)
            foref_pred[2] = prediction_angle  # the prediction angle here is NOT normalized already.

            foref_true = read_foref(dataset, torch.cat([batch[FdFoRefOrigin.NAME],
                                                        batch[FdFoRefAngle.NAME]], 1)[0])
            # Record results
            results["true_pred_angle_diff"].append(
                math.degrees(clamp_angle(abs(foref_true[2] - foref_pred[2]))))
            results["true_pred_origin_diff"].append(
                euclidean_dist(foref_true[:2], foref_pred[:2]))

            # Create example plots
            if i in plot_indices:
                mapimg = batch[FdCtxImg.NAME][0].numpy().reshape(map_dims)
                mapimg = dataset.rescale(FdCtxImg.NAME, mapimg)
                objloc = dataset.rescale(FdAbsObjLoc.NAME, batch[FdAbsObjLoc.NAME][0])

                fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5, 5))
                colors = {"prediction": ("coral", "yellowgreen"),
                          "true": ("purple", "darkkhaki")}

                plot_multiple(mapimg,
                              {"prediction": foref_pred,
                               "true": foref_true},
                              objloc,
                              colors, ax,
                              map_dims=map_dims)
                plt.title("%s" % (keyword))
                plt.savefig(os.path.join(plots_dir, "%s-%s-%d.png" % (keyword, suffix, i+1)))
                plt.clf()
            sys.stdout.write("Computing metrics and plotting ...[%d/%d]\r" % (i+1, len(data_loader)))

        # Plot 1d plots -- These are summary plots
        plot_1d(results["true_pred_angle_diff"], "Angle differences between true and prdicted FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-true_pred_angle_diff.png" % (keyword, suffix)))
        plt.clf()

        plot_1d(results["true_pred_origin_diff"], "Distances between true and predicted FoRs")
        plt.savefig(os.path.join(plots_dir, "%s-%s-true_pred_origin_diff.png" % (keyword, suffix)))
        plt.clf()

        titles = {
            "true_pred_angle_diff": "Angle differences between true and predicted FoRs",
            "true_pred_origin_diff": "Distances between true and predicted FoRs",
        }

        # Save metrics
        results["__summary__"] = {}
        for catg in results:
            if catg.startswith("__"):
                continue
            plot_1d(results[catg], titles[catg])
            plt.savefig(os.path.join(plots_dir, "%s-%s-%s.png" % (keyword, suffix, catg)))
            plt.clf()

            mean, ci = mean_ci_normal(results[catg], confidence_interval=0.95)
            results["__summary__"][catg] = {
                "mean": mean,
                "ci-95": ci
            }
        with open(os.path.join(metrics_dir, "foref_deviation.json"), "w") as f:
            json.dump(json_safe(results), f, indent=4, sort_keys=True)

        print("Summary results:")
        pprint(results["__summary__"])
        plt.close()
