import os
import torch
from datetime import datetime as dt
import json
import time
import yaml
import pandas as pd

from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.utils import json_safe, smart_json_hook
from sloop.models.nn.all_models import *
from sloop.datasets.dataloader import FdMapName, SpatialRelationDataset

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


def get_model_class(model_type, iteration=1):
    if model_type in MODELS_BY_ITER[iteration]:
        return MODELS_BY_ITER[iteration][model_type]
    else:
        raise ValueError("Model %s does not appear in ITERATION %d" % (model_type, iteration))

## Plotting

def plot_losses(keyword, train_losses, val_losses, model_type, log_path, suffix=""):
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss for %s" % keyword)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "loss-%s%s.png" % (keyword, suffix)))
    plt.clf()

def evaluate(model, config, data_kind, dataset, args,
             model_save_dirpath, device="cpu"):
    print("Saving %s data's dataframe to %s/%s_df.pkl.zip"
          % (data_kind, model_save_dirpath, data_kind))
    dataset.df.to_pickle(os.path.join(model_save_dirpath,
                                      "%s_df.pkl.zip" % data_kind))


    # Print reproduce command first.
    # Completely reproducible results are not guaranteed across PyTorch releases,
    # individual commits or different platforms. Furthermore, results need not be
    # reproducible between CPU and GPU executions, even when using identical seeds.
    # However, in order to make computations deterministic on your specific problem
    # on one specific platform and PyTorch release, there are a couple of steps to
    # take.
    arguments = [args.keyword,
                 ",".join(dataset.map_names),
                 str(args.iteration),
                 args.model_type,
                 os.path.join(model_save_dirpath, "%s_model.pt" % args.keyword),
                 os.path.join(model_save_dirpath, "%s_df.pkl.zip" % data_kind),
                 data_kind,
                 os.path.join(model_save_dirpath, "config.yaml"),
                 os.path.join(model_save_dirpath, "train_meta.json"),
                 "-o", os.path.join(model_save_dirpath, "reproduce_%s" % data_kind)]

    reproduce_command = "python reproduce.py " + " ".join(arguments)
    with open(os.path.join(model_save_dirpath, "reproduce_%s.txt" % data_kind), "w") as f:
        f.write(reproduce_command)
        print("\nCommand to reproduce %s results:" % data_kind)
        print(reproduce_command)


    # The Plot function can create plots & save metrics.
    print("~~~ Evaluating on %s dataset ~~~" % data_kind)
    model.__class__.Eval(args.keyword, model, dataset,
                         device, model_save_dirpath,
                         suffix=data_kind,
                         map_dims=config["data_params"]["shared"]["desired_dims"],
                         **config["eval_params"])
    model.__class__.Plot(args.keyword, model, dataset,
                         device, model_save_dirpath,
                         suffix=data_kind,
                         map_dims=config["data_params"]["shared"]["desired_dims"],
                         **config["plot_params"])


## Main
def main():
    parser = argparse.ArgumentParser(description="Process spatial graph json files.")
    parser.add_argument("-dRaw", "--datadir", type=str, help="Path to directory pickle data files")
    parser.add_argument("-dTrain", "--train-data", type=str,
                        help="Path to data frame pickle file for training set")
    parser.add_argument("-dVal", "--val-data", type=str,
                        help="Path to data frame pickle file for validation set")
    parser.add_argument("-dTest", "--test-data", type=str,
                        help="Path to data frame pickle file for test set")
    parser.add_argument("keyword", type=str,
                        help="keywords to train for")
    parser.add_argument("iteration", type=int,
                        help="Iteration number in the research process. E.g. 1,2,3 etc.")
    parser.add_argument("model_type", type=str,
                        help="Type of model to train.")
    parser.add_argument("config_file", type=str,
                        help="Path to .yaml configuration file")
    parser.add_argument("--log-path", type=str, default="./logs",
                        help="Root path to store results for this particular type of model")
    parser.add_argument("--cuda", type=str, default="0",
                        help="Cuda to use, if available.")
    parser.add_argument("--config-change", type=str,
                        help="A JSON formatted string, with a list of changes;"
                        "Updates the config in the provided config file by the values in this string")


    args = parser.parse_args()

    test_map = os.path.splitext(os.path.basename(args.config_file))[0].split("-")[1]
    device = torch.device("cuda:%s" % args.cuda if torch.cuda.is_available() else "cpu")
    start_time = dt.now()
    timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]

    ## Set up directory to save logs
    log_path = os.path.join(args.log_path,
                            "iter_%d" % args.iteration,
                            args.model_type,
                            args.keyword,
                            "iter%d_%s:%s:%s_%s" % (args.iteration,
                                                    args.model_type.replace("_","-"),
                                                    args.keyword,
                                                    test_map.replace("_",","),
                                                    timestr))
    model_save_dirpath = log_path
    if not os.path.exists(model_save_dirpath):
        os.makedirs(model_save_dirpath)

    ## Obtaining configurations
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
        if args.config_change:
            config_change = json.loads(args.config_change,
                                       object_hook=smart_json_hook)
            for chg in config_change:
                cur_config_dict = config
                new_val = config_change[chg]
                keys = chg.split(":")
                for k in keys[:-1]:
                    cur_config_dict = cur_config_dict[k]
                cur_config_dict[keys[-1]] = new_val
    print("All configs:")
    pprint(config)
    print("\nSaving training configurations to %s/params.json" % model_save_dirpath)
    args_dct = vars(args)
    with open(os.path.join(model_save_dirpath, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    ## Getting the model
    model_class = get_model_class(args.model_type, args.iteration)
    model = model_class(args.keyword,
                        learning_rate=float(config["model_params"]["learning_rate"]),
                        map_dims=config["data_params"]["shared"]["desired_dims"])
    count_parameters(model)
    train_func = model_class.Train

    ## Getting the Training data
    train_dataset = None
    if args.train_data:
        train_df = pd.read_pickle(args.train_data)
        map_names = set()
        for index, row in train_df.iterrows():
            map_names.add(row[FdMapName.NAME])
        train_dataset = SpatialRelationDataset(train_df, map_names)
        train_metadata = {"_normalizers_": train_dataset.normalizers}
        train_set_loaded = True
    elif args.datadir:
        print("\n###Preparing training data for \"%s\"... ###" % args.keyword)
        train_dataset, train_metadata = model_class.get_data(
            args.keyword, args.datadir, config["train_maps"],
            **{**config["data_params"]["train"], **config["data_params"]["shared"]})
        train_metadata["_normalizers_"] = train_dataset.normalizers
        # Provide the model the normalizers for this dataset. Needed when loading
        # the model back from a file to evaluate the model.
    if train_dataset is not None:
        model.normalizers = train_dataset.normalizers
        print("----Train data normalizers: ")
        pprint(train_dataset.normalizers)
        print("----Total number of training samples: %d" % len(train_dataset.df))
        time.sleep(0.2)

        print("Saving training data metadata to %s/train_meta.json" % model_save_dirpath)
        with open(os.path.join(model_save_dirpath, "train_meta.json"), "w") as f:
            json.dump(json_safe(train_metadata), f, indent=4, sort_keys=True)

        print("Saving training dataframe to %s/train_df.pkl.zip" % model_save_dirpath)
        train_dataset.df.to_pickle(os.path.join(model_save_dirpath, "train_df.pkl.zip"))

    test_set_loaded = False
    test_dataset = None
    if args.test_data:
        test_df = pd.read_pickle(args.test_data)
        map_names = set()
        for index, row in test_df.iterrows():
            map_names.add(row[FdMapName.NAME])
        test_dataset = SpatialRelationDataset(test_df, map_names)
        test_set_loaded = True
    elif len(config["test_maps"]) > 0:
        print("\n### Preparing test data for %s... ###" % args.keyword)
        test_dataset, _ = model_class.get_data(
            args.keyword, args.datadir, config["test_maps"],
            normalizers=train_dataset.normalizers,
            **{**config["data_params"]["test"], **config["data_params"]["shared"]})
    if test_dataset is not None:
        print("    Test data normalizers: ")
        pprint(test_dataset.normalizers)
        print("    Total number of test samples: %d" % len(test_dataset.df))
        print("Saving test data's dataframe to %s/test_df.pkl.zip" % model_save_dirpath)
        test_dataset.df.to_pickle(os.path.join(model_save_dirpath, "test_df.pkl.zip"))
        time.sleep(0.2)

    ## Getting the validation data
    val_set_loaded = False
    val_dataset = None
    if args.val_data:
        val_df = pd.read_pickle(args.val_data)
        map_names = set()
        for index, row in val_df.iterrows():
            map_names.add(row[FdMapName.NAME])
        val_dataset = SpatialRelationDataset(val_df, map_names,
                                             normalizers=train_dataset.normalizers)
        val_set_loaded = True
    elif len(config["val_maps"]) > 0:
        print("\n### Preparing validation data for %s... ###" % args.keyword)
        val_dataset, _ = model_class.get_data(
            args.keyword, args.datadir, config["val_maps"],
            normalizers=train_dataset.normalizers,
            **{**config["data_params"]["val"], **config["data_params"]["shared"]})
    if val_dataset is not None:
        print("    Validation data normalizers: ")
        pprint(val_dataset.normalizers)
        print("    Total number of validation samples: %d" % len(val_dataset.df))
        time.sleep(0.2)

    ## Training, and save necessary information
    model.to(device)

    #### Actual training starts ####
    print("\nTraining...")

    res = train_func(model, train_dataset, device,
                     valset=val_dataset,
                     save_dirpath=model_save_dirpath,
                     **config["learning_params"])

    # Saving, and evaluate
    model.eval()
    if res is not None:
        train_losses, val_losses, train_dataset, val_dataset = res

        print("Saving training data's dataframe to %s/train_df.pkl.zip" % model_save_dirpath)
        train_dataset.df.to_pickle(os.path.join(model_save_dirpath, "train_df.pkl.zip"))

        print("Saving validation data's dataframe to %s/val_df.pkl.zip" % model_save_dirpath)
        val_dataset.df.to_pickle(os.path.join(model_save_dirpath, "val_df.pkl.zip"))

        print("\nSaving Additional training data metadata to %s/train_log.json" % model_save_dirpath)
        meta = {"training_data_size": len(train_dataset),
                "validation_data_size": len(val_dataset),
                "train_losses": train_losses,
                "val_losses": val_losses}
        with open(os.path.join(model_save_dirpath, "train_log.json"), "w") as f:
            json.dump(json_safe(meta), f, indent=4, sort_keys=True)

        ## Plot loss
        if type(train_losses) == list:
            plot_losses(args.keyword,
                        train_losses,
                        val_losses,
                        args.model_type,
                        log_path)
        else:
            for module_name in train_losses:
                plot_losses(args.keyword,
                            train_losses[module_name],
                            val_losses[module_name],
                            args.model_type,
                            log_path,
                            suffix="-" + module_name)
        ## Evaluate
        evaluate(model, config, "train", train_dataset, args,
                 model_save_dirpath, device=device)
        evaluate(model, config, "val", val_dataset, args,
                 model_save_dirpath, device=device)

    if len(config["test_maps"]) > 0:
        evaluate(model, config, "test", test_dataset, args,
                 model_save_dirpath, device=device)

if __name__ == "__main__":
    main()
