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
from sloop.models.nn.iter2.common import *
import numpy as np
import json
from pprint import pprint

class CtxForefAngleModel(BaseModel):
    NAME="ctx_foref_angle"

    def __init__(self, keyword, learning_rate=1e-4, map_dims=(21,21)):
        super(CtxForefAngleModel, self).__init__()
        self.keyword = keyword
        self.map_dims = map_dims
        self.map_layer = MapImgCNN(map_dims=map_dims)
        self.layer_forigin_input = nn.Sequential(nn.Linear(2, 16),
                                                 nn.ReLU(),
                                                 nn.Linear(16, 16),
                                                 nn.ReLU(),
                                                 nn.Linear(16, 16))
        featdim = {
            (41,41): 768,
            (28,28): 300
        }
        self.layer_foref = nn.Sequential(nn.Linear(featdim[tuple(map_dims)], FOREF_L1),
                                         nn.ReLU(),
                                         nn.Linear(FOREF_L1, FOREF_L2),
                                         nn.ReLU(),
                                         nn.Linear(FOREF_L2, FOREF_L3),
                                         nn.ReLU(),
                                         nn.Linear(FOREF_L3, 1))
        self.input_size = map_dims[0]*map_dims[1]
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.normalizers = {}

    def forward(self, x):
        x_map = self.map_layer(x)
        out = self.layer_foref(x_map)
        return out

    @classmethod
    def Input_Fields(cls):
        return [FdCtxImg.NAME]

    @classmethod
    def Output_Fields(cls):
        return [FdFoRefAngle.NAME]

    @classmethod
    def get_data(cls, keyword, data_dirpath, map_names,
                 **kwargs):
        return get_common_data(keyword, data_dirpath, map_names, **kwargs)

    @classmethod
    def Train(cls, model, trainset, device,
              **kwargs):
        # First, train the frame of reference module
        print("Training the frame of reference module")
        criterion = FoRefAngleLoss(trainset, device=device)
        return super(CtxForefAngleModel, cls).Train(model, trainset, device,
                                                      criterion=criterion,
                                                      **kwargs)

    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo,
                   as_batch=True, **kwargs):
        inpt = torch.cat([data_sample[FdCtxImg.NAME].float()], 1)
        return inpt.reshape(1,-1)

    def predict_foref(self, keyword, landmark_symbol,
                      map_name, mapinfo, device="cpu", map_img=None):
        if keyword != self.keyword:
            print("Given keyword %s != model's keyword %s" % (keyword, self.keyword))
            return None
        if len(self.normalizers) == 0:
            raise ValueError("Normalizers not present.")

        dummy_dataset = SpatialRelationDataset(None, None,
                                               normalizers=self.normalizers)

        # create context image
        if map_img is None:
            ctx_img = make_context_img(mapinfo, map_name, landmark_symbol, dist_factor=2.0).flatten()
        else:
            ctx_img = map_img
        ctx_img = dummy_dataset.normalize(FdCtxImg.NAME, ctx_img)
        prediction = self(torch.tensor(ctx_img).reshape(1,-1).float().to(device))[0]
        foref = read_foref(dummy_dataset, prediction)
        return foref

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
            inpt = cls.make_input(batch, dataset, mapinfo, as_batch=True)
            prediction_angle = model(inpt.to(device))[0].item()

            foref_pred = np.array([*batch[FdFoRefOrigin.NAME][0].numpy(),
                                   prediction_angle])
            foref_pred = read_foref(dataset, foref_pred)

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

                # If the keyword is "front", plot the frame of reference as is.
                # If it is left, rotate it by -90 degrees (because the prediction is front,
                # and we want to plot the left direction). Only plot that vector
                if keyword == "front":
                    foref_pred_plot = foref_pred
                    foref_true_plot = foref_true
                    colors = {"prediction": ("salmon", "gold"),
                              "true": ("red", "darkkhaki")}
                elif keyword == "left":
                    left_angle_pred = foref_pred[2] + math.radians(-90)
                    foref_pred_plot = [foref_pred[0], foref_pred[1], left_angle_pred]
                    left_angle_true = foref_true[2] + math.radians(-90)
                    foref_true_plot = [foref_true[0], foref_true[1], left_angle_true]
                    colors = {"prediction": ("lime", "orange"),
                              "true": ("limegreen", "darkkhaki")}

                plot_multiple(mapimg,
                              {"prediction": foref_pred_plot,
                               "true": foref_true_plot},
                              objloc,
                              colors, ax,
                              map_dims=map_dims,
                              width_factor=3.0,
                              plot_perp=False,
                              plot_obj=False)
                ax.axis("off")
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
