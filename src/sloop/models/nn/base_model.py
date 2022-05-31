import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sloop.datasets.dataloader import *
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
from sloop.models.nn.loss_function import FoRefLoss, clamp_angle
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import json
import random


class BaseModel(nn.Module):

    @classmethod
    def Train(cls, model, trainset, device,
              val_ratio=0.2, num_epochs=500, batch_size=10, shuffle=True,
              save_dirpath=None, loss_threshold=1e-4, early_stopping=False,
              valset=None, window=20, criterion=nn.MSELoss(reduction="sum"),
              input_fields=None, output_fields=None, model_args={}):

        """
        model (nn.Module): The network
        trainset (Dataset)
        val_ratio (float) ratio of the trainset to use as validation set
        loss_threshold (float): The threshold of training loss change.
        early_stopping (bool): True if terminate when validation loss increases,
            by looking at the average over the previous 10 epochs.
        """
        assert type(trainset) == SpatialRelationDataset

        if valset is None:
            print("Splitting training set by 1:%.2f to get validation set." % val_ratio)
            trainset, valset = trainset.split(val_ratio)
        print("Train set size: %d" % len(trainset))
        print("Validation set size: %d" % len(valset))
        print(device)

        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=shuffle)

        train_losses = []
        val_losses = []
        try:
            for epoch in range(num_epochs):
                running_train_loss = 0.0

                batch_num = 0
                for batch in train_loader:  # iterate by batches
                    # Must transform to float because that's what pytorch supports.
                    input_data = []
                    if input_fields is None:
                        input_fields = cls.Input_Fields()
                    for fd_name in input_fields:
                        input_data.append(batch[fd_name].float())
                    train_inputs = torch.cat(input_data, 1).float()

                    output_data = []
                    if output_fields is None:
                        output_fields = cls.Output_Fields()
                    for fd_name in output_fields:
                        output_data.append(batch[fd_name].float())
                    train_labels = torch.cat(output_data, 1).float()
                    model.train()  # Training mode
                    prediction = model(train_inputs.to(device), **model_args)
                    train_loss = criterion(prediction, train_labels.to(device))
                    model.zero_grad()
                    train_loss.backward()
                    model.optimizer.step()

                    running_train_loss += train_loss.item()
                    batch_num += 1

                # Record losses per epoch; Both training and validation losses
                print('[%d] loss: %.5f' %
                      (epoch + 1, running_train_loss / batch_num))

                train_losses.append(running_train_loss / batch_num)
                # Is the loss converging?
                train_done = False
                if loss_threshold is not None and epoch % window == 0:
                    if len(train_losses) >= window * 2:
                        t_now = np.mean(train_losses[-window:])
                        t_prev = np.mean(train_losses[-2*window:-window])
                        loss_diff = abs(t_now - t_prev)
                        if loss_diff < loss_threshold:
                            train_done = True
                            print("Training loss converged.")

                # Compute validation loss
                running_val_loss = 0.0
                batch_num = 0
                for batch in val_loader:
                    input_data = []

                    if input_fields is None:
                        input_fields = cls.Input_Fields()
                    for fd_name in input_fields:
                        input_data.append(batch[fd_name].float())
                    val_inputs = torch.cat(input_data, 1).float()

                    output_data = []
                    if output_fields is None:
                        output_fields = cls.Output_Fields()
                    for fd_name in output_fields:
                        output_data.append(batch[fd_name].float())
                    val_labels = torch.cat(output_data, 1).float()
                    prediction = model(val_inputs.to(device), **model_args)
                    val_loss = criterion(prediction, val_labels.to(device))
                    running_val_loss += val_loss.item()
                    batch_num += 1
                val_losses.append(running_val_loss / batch_num)

                if early_stopping:
                    if epoch % window == 0:
                        if len(val_losses) >= window*2:
                            v_now = np.mean(val_losses[-window:])
                            v_prev = np.mean(val_losses[-2*window:-window])
                            if v_now > v_prev:
                                # Validation loss increased. Stop
                                print("Validation loss incrased (window size = %d). Stop." % window)
                                train_done = True

                if train_done:
                    break
        except KeyboardInterrupt:
            print("Training interrupted.")
        except Exception as ex:
            raise ex
        if save_dirpath is not None:
            if not os.path.exists(save_dirpath):
                os.makedirs(save_dirpath)
            torch.save(model, os.path.join(save_dirpath, model.keyword + "_model.pt"))
        return train_losses, val_losses, trainset, valset


    @classmethod
    def compute_ops(cls,
                    mapinfo,
                    keyword=None,
                    augment_radius=0,
                    augment_dfactor=None,
                    fill_neg=False,
                    rotate_amount=0,
                    rotate_range=(0,360),
                    translate_amount=0,
                    add_pr_noise=0.15):
        data_ops = []
        if augment_radius > 0:
            op = (OpAugPositive,
                  (mapinfo, augment_radius),
                  {"dfactor": augment_dfactor})
            data_ops.append(op)
        if fill_neg:
            op = (OpFillNeg,
                  (mapinfo, keyword))
            data_ops.append(op)

        op_add_noise = (OpProbAddNoise,
                         tuple(),
                         {"noise": add_pr_noise})
        data_ops.append(op_add_noise)

        if rotate_amount > 0:
            op = (OpRandomRotate, (mapinfo,), {"amount": rotate_amount,
                                               "rotate_range": rotate_range})
            data_ops.append(op)
        if translate_amount > 0:
            op = (OpRandomTranslate, (mapinfo,), {"amount": translate_amount})
            data_ops.append(op)
        return data_ops

    @classmethod
    def make_input(cls, data_sample, dataset, mapinfo, as_batch=True, abs_obj_loc=None, **kwargs):
        raise NotImplemented

    @classmethod
    def compute_heatmap(cls, model, data_sample,
                        dataset, mapinfo, device="cpu", map_dims=None):
        """
        data_sample is a row in the dataset. It contains fields
        such as frame of reference, and the map image. We aim to
        compute a heatmap on top of these information.
        """
        landmark_symbol = data_sample[FdLmSym.NAME]
        map_name = mapinfo.map_name_of(landmark_symbol)
        if map_dims is None:
            map_dims = mapinfo.map_dims(map_name)
        heatmap = np.zeros(map_dims, dtype=np.float64)

        all_inputs = []
        for idx in range(map_dims[0]*map_dims[1]):
            x = idx // map_dims[1]
            y = idx - x * map_dims[1]
            inpt = cls.make_input(data_sample, dataset, mapinfo,
                                  as_batch=True,
                                  abs_obj_loc=(x,y))
            all_inputs.append(inpt)
        # We are concatenating the tensors by rows.
        all_inputs = torch.cat(all_inputs, 0)
        prediction = model(all_inputs.to(device))

        total_prob = 0.0
        for idx in range(map_dims[0]*map_dims[1]):
            x = idx // map_dims[1]
            y = idx - x * map_dims[1]
            prob = dataset.rescale(FdProbSR.NAME, prediction[idx].item())
            heatmap[x,y] = prob
            total_prob += prob
        print(np.unique(prediction.cpu().data.numpy()))
        heatmap[x,y] /= total_prob
        return heatmap

    @classmethod
    def Eval_OutputPr(cls, keyword, model, dataset, device, save_dirpath,
             suffix="group", **kwargs):
        """Evaluation of models that output spatial relation probability"""
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())

        metrics_dir = os.path.join(save_dirpath, "metrics", suffix)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        results = {"perplex_true": [],  # The perplexity of a distribution for the true object location
                   "perplex_pred": [],  # The perplexity of the predicted heatmap
                   "kl_div": [],     # The kl divergence between true and predicted distributions
                   "distance": []}  # The distance between most likely object location and true object location

        map_dims = kwargs.get("map_dims", (21,21))
        all_locations = [(x,y)
                         for x in range(map_dims[0])
                         for y in range(map_dims[1])]
        variance = [[1, 0], [0,1]]
        amount = min(len(dataset), kwargs.get("test_amount", 100))
        indices = list(np.arange(len(dataset)))
        random.Random(100).shuffle(indices)
        for kk in range(amount):
            i = indices[kk]
            data_sample = dataset[i]  # The returned data_sample should be normalized
            objloc = dataset.rescale(FdAbsObjLoc.NAME,
                                     data_sample[FdAbsObjLoc.NAME])
            true_dist = normal_pdf_2d(objloc, variance, all_locations)
            heatmap = cls.compute_heatmap(model, data_sample,
                                          dataset, mapinfo, device=device,
                                          map_dims=map_dims)
            pred_dist = {(x,y): heatmap[x,y]
                         for x,y in all_locations}
            # Convert the dictionary distributions into sequences, with matching
            # elements at each index.
            seqs, vals = dists_to_seqs([true_dist, pred_dist], avoid_zero=True)
            # Compute metrics and record
            perplex_true = perplexity(seqs[0])
            perplex_pred = perplexity(seqs[1])
            kl_div = kl_divergence(seqs[0], q=seqs[1])
            results["perplex_true"].append(perplex_true)
            results["perplex_pred"].append(perplex_pred)
            results["kl_div"].append(kl_div)

            objloc_pred = max(pred_dist, key=lambda x: pred_dist[x])
            dist = euclidean_dist(objloc_pred, objloc)
            results["distance"].append(dist)

            sys.stdout.write("Computing heatmaps & metrics...[%d/%d]\r" % (kk+1, amount))

        results = compute_mean_ci(results)
        with open(os.path.join(metrics_dir, "information_metrics.json"), "w") as f:
            json.dump(json_safe(results), f, indent=4, sort_keys=True)

        print("Summary results:")
        pprint(results["__summary__"])


    @classmethod
    def Plot_OutputPr(cls, keyword, model, dataset, device, save_dirpath,
             suffix="plot", **kwargs):
        """Plot some examples for the model whose output is spatial relation probability"""
        mapinfo = MapInfoDataset()
        for map_name in dataset.map_names:
            mapinfo.load_by_name(map_name.strip())
        amount = kwargs.get("plot_amount", 10)
        plotsdir = os.path.join(save_dirpath, "plots", suffix)
        if not os.path.exists(plotsdir):
            os.makedirs(plotsdir)

        indices = list(np.arange(len(dataset)))
        random.Random(100).shuffle(indices)
        all_inputs = []
        for kk in range(min(amount, len(indices))):
            i = indices[kk]
            data_sample = dataset[i]
            inpt = cls.make_input(data_sample, dataset, mapinfo,
                                  as_batch=True)
            all_inputs.append(inpt)
        all_inputs = torch.cat(all_inputs, 0)
        prediction = model(all_inputs.to(device))

        # Make plots
        map_dims = kwargs.get("map_dims", None)
        for i in range(len(prediction)):
            data_sample = dataset[i]
            pred_prob = dataset.rescale(FdProbSR.NAME, prediction[i].item())
            true_prob = dataset.rescale(FdProbSR.NAME, data_sample[FdProbSR.NAME])
            heatmap = cls.compute_heatmap(model, data_sample,
                                          dataset, mapinfo, device=device,
                                          map_dims=map_dims)

            # Plot map
            ax = plt.gca()
            if FdBdgImg.NAME in data_sample:
                mapimg = data_sample[FdBdgImg.NAME].reshape(map_dims)
                mapimg = dataset.rescale(FdBdgImg.NAME, mapimg)
            elif FdCtxImg.NAME in data_sample:
                mapimg = data_sample[FdCtxImg.NAME].reshape(map_dims)
                mapimg = dataset.rescale(FdCtxImg.NAME, mapimg)
            elif FdMapImg.NAME in data_sample:
                mapimg = data_sample[FdMapImg.NAME].reshape(map_dims)
                mapimg = dataset.rescale(FdMapImg.NAME, mapimg)
            else:
                continue
            plot_map(ax, mapimg.numpy().transpose())
            # Plot heatmap
            plot_map(ax, heatmap, alpha=0.6)

            # Plot object location
            objloc = dataset.rescale(FdAbsObjLoc.NAME, data_sample[FdAbsObjLoc.NAME])
            ax.scatter([objloc[0].item()], [objloc[1].item()], s=100, c="cyan")

            # Plot frame of reference
            if FdFoRefOrigin.NAME in data_sample:
                foref_true = read_foref(dataset,
                                        [*data_sample[FdFoRefOrigin.NAME],
                                         data_sample[FdFoRefAngle.NAME]])
                plot_foref(foref_true, ax, c1="magenta", c2="lime")

            plt.title("%s : %.3f" % (keyword, pred_prob))
            plt.savefig(os.path.join(plotsdir, "%s-%s-%d.png" % (keyword, suffix, i+1)))
            plt.clf()
            sys.stdout.write("Plotting ...[%d/%d]\r" % (i+1, len(prediction)))


    @classmethod
    def Eval_OutputForef(cls, keyword, model, dataset, device,
                         save_dirpath, suffix="metrics", **kwargs):
        print("Eval for class %s is done together through Plot." % cls.__name__)

    @classmethod
    def Plot_OutputForef(cls, keyword, model, dataset, device,
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
            # train_inputs = torch.cat([
            #     batch[FdProbSR.NAME].float(),
            #     batch[FdBdgImg.NAME].float(),
            #     batch[FdObjLoc.NAME].float(),
            #     batch[FdBdgEgoImg.NAME].float(),
            #     batch[FdAbsObjLoc.NAME].float(),
            # ], 1).float()
            prediction_data = model(inpt.to(device))[0]
            if relative_origin:
                # foref_pred = read_foref(dataset, prediction_data,
                #                         relative=relative_origin)
                # foref_true = read_foref(dataset, torch.cat([batch[FdFoRefOriginRel.NAME],
                #                                             batch[FdFoRefAngle.NAME]], 1)[0],
                #                         relative=relative_origin)
                # # Convert origin to absolute coordinates
                # landmark = batch[FdLmSym.NAME][0]
                # map_name = batch[FdMapName.NAME][0]
                # lmk_ctr = mapinfo.center_of_mass(landmark, map_name)
                # foref_pred[0] += lmk_ctr[0]
                # foref_pred[1] += lmk_ctr[1]
                # foref_true[0] += lmk_ctr[0]
                # foref_true[1] += lmk_ctr[1]
                print("THIS DOES NOT WORK.")
            else:
                foref_pred = read_foref(dataset, prediction_data)
                foref_true = read_foref(dataset, torch.cat([batch[FdFoRefOrigin.NAME],
                                                            batch[FdFoRefAngle.NAME]], 1)[0])

            # Record results
            results["true_pred_angle_diff"].append(
                math.degrees(clamp_angle(abs(foref_true[2] - foref_pred[2]))))
            results["true_pred_origin_diff"].append(
                euclidean_dist(foref_true[:2], foref_pred[:2]))

            # Create example plots
            if i in plot_indices:
                if FdBdgImg.NAME in batch:
                    mapimg = batch[FdBdgImg.NAME][0].numpy().reshape(map_dims)
                    mapimg = dataset.rescale(FdBdgImg.NAME, mapimg)
                elif FdCtxImg.NAME in batch:
                    mapimg = batch[FdCtxImg.NAME][0].numpy().reshape(map_dims)
                    mapimg = dataset.rescale(FdCtxImg.NAME, mapimg)
                elif FdMapImg.NAME in batch:
                    mapimg = batch[FdMapImg.NAME][0].numpy().reshape(map_dims)
                    mapimg = dataset.rescale(FdMapImg.NAME, mapimg)
                else:
                    continue
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
