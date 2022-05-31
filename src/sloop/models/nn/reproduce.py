# Script used to reproduce plots / other metric results by
# loading a saved model.

import os
import json
import yaml
import numpy as np
import torch
import argparse
import pandas as pd
from sloop.models.nn.train_model import get_model_class
from sloop.datasets.dataloader import SpatialRelationDataset
from pprint import pprint

def cli():
    parser = argparse.ArgumentParser(description="Reproduce results")

    parser.add_argument("keyword", type=str,
                        help="The keyword the model was for")
    parser.add_argument("map_names", type=str,
                        help="comma separated list of map names whose data is involved")
    parser.add_argument("iteration", type=int,
                        help="Iteration number in the research process. E.g. 1,2,3 etc.")
    parser.add_argument("model_type", type=str,
                        help="type of model. e.g. pr_to_fref")
    parser.add_argument("model_path", type=str,
                        help="Path to model .pt file")
    parser.add_argument("df_path", type=str,
                        help="Path to dataframe (pkl file)")
    parser.add_argument("data_kind", type=str,
                        help="is the data train, validation, or test?")
    parser.add_argument("config_path", type=str,
                        help="Path to config yaml file")
    parser.add_argument("train_metadata_path", type=str,
                        help="Path to train_meta.json file.")
    parser.add_argument("-o", "--output-dir", type=str, default="./reproduce",
                        help="Directory to output ")
    parser.add_argument("--cuda", type=str, default="0",
                        help="Cuda to use, if available.")
    args = parser.parse_args()
    return args

def reproduce(args):
    # Completely reproducible results are not guaranteed across PyTorch releases,
    # individual commits or different platforms. Furthermore, results need not be
    # reproducible between CPU and GPU executions, even when using identical seeds.
    # However, in order to make computations deterministic on your specific problem
    # on one specific platform and PyTorch release, there are a couple of steps to
    # take.
    device = torch.device("cuda:%s" % args.cuda if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model_class = get_model_class(args.model_type, iteration=args.iteration)
    model = torch.load(args.model_path, map_location=device)
    assert model.keyword == args.keyword
    model.eval()

    map_names = args.map_names.split(",")
    for i in range(len(map_names)):
        map_names[i] = map_names[i].strip()

    print("Loading training metadata")
    with open(os.path.join(args.train_metadata_path)) as f:
        normalizers = json.load(f)["_normalizers_"]
        for fd in normalizers:
            normalizers[fd] = tuple(np.array(normalizers[fd]).astype(float))
    print("Normalizers:")
    pprint(normalizers)

    print("Loading data...")
    df = pd.read_pickle(args.df_path)
    dataset = SpatialRelationDataset(df, map_names, normalizers=normalizers)

    print("Reproducing results...")
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    model_class.Eval(args.keyword, model, dataset, device, args.output_dir,
                     suffix=args.data_kind,
                     map_dims=config["data_params"]["shared"]["desired_dims"],
                     **config["eval_params"])
    model_class.Plot(args.keyword, model, dataset, device, args.output_dir,
                     suffix=args.data_kind,
                     map_dims=config["data_params"]["shared"]["desired_dims"],
                     **config["plot_params"])

if __name__ == "__main__":
    args = cli()
    reproduce(args)
