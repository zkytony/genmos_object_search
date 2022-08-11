#
import os
import itertools
import numpy as np
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.datasets.dataloader import *
from sloop.datasets.utils import *
from sloop.models.nn.plotting import *
from sloop.models.nn.loss_function import FoRefLoss, clamp_angle

datadir = "../../datasets/SL-OSM-Dataset/not_blind/FOR_only/sg_processed"
outdir = "./data/not_blind"
maps = ["cleveland", "denver", "austin", "honolulu", "washington_dc"]
mapinfo = MapInfoDataset()
for map_name in maps:
    mapinfo.load_by_name(map_name)

os.makedirs(outdir, exist_ok=True)

desired_dims=(28,28)
fields = [(FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
          (FdCtxImg, (mapinfo,), {"desired_dims": desired_dims,
                                  "use_nbr": True}),
          (FdCtxEgoImg, (mapinfo,), {"desired_dims": desired_dims}),
          (FdBdgEgoImg, (mapinfo,), {"desired_dims": desired_dims}),
          (FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
          (FdFoRefAngle, tuple()),
          (FdLmSym, tuple()),
          (FdMapName, tuple())]
for keyword in {"front", "left"}:
    print(keyword)
    for test_map in maps:
        print("  %s" % test_map)
        dataset = SpatialRelationDataset.build(keyword,
                                               [test_map], datadir,
                                               fields=fields)
        pp = os.path.join(outdir, keyword, test_map)
        os.makedirs(pp, exist_ok=True)
        dataset.df.to_pickle(os.path.join(pp, "test_df.pkl.zip"))
