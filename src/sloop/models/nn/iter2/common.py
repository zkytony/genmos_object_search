import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sloop.models.nn.base_model import BaseModel
from sloop.datasets.dataloader import *
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset

# ROUND 1 PARAMS: does not generalize well
CONV1_PLAIN = 6
CONV1_KERNEL = 5
CONV2_PLAIN = 12
CONV2_KERNEL = 3
CONV3_PLAIN = 24
CONV3_KERNEL = 3
FOREF_L1 = 128
FOREF_L2 = 64
FOREF_L3 = 32

class MapImgCNN(nn.Module):
    def __init__(self, map_dims=(21,21)):
        super(MapImgCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, CONV1_PLAIN,
                               kernel_size=CONV1_KERNEL,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(CONV1_PLAIN)
        self.max_pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(CONV1_PLAIN, CONV2_PLAIN,
                               kernel_size=CONV2_KERNEL, bias=False)
        self.max_pool2 = nn.MaxPool2d((2,2))
        self.bn2 = nn.BatchNorm2d(CONV2_PLAIN)

        self.conv3 = nn.Conv2d(CONV2_PLAIN, CONV3_PLAIN,
                               kernel_size=CONV3_KERNEL, bias=False)
        self.max_pool3 = nn.MaxPool2d((2,2))
        self.bn3 = nn.BatchNorm2d(CONV3_PLAIN)

        self.input_size = map_dims[0]*map_dims[1]
        self.map_dims = map_dims

    def forward(self, x):
        x = x.view(-1, 1, self.map_dims[0], self.map_dims[1])
        out = self.max_pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.max_pool2(F.relu(self.bn2(self.conv2(out))))
        out = out.view(out.shape[0], -1)
        return out



def get_common_data(keyword, data_dirpath, map_names,
                    augment_radius=0,
                    augment_dfactor=None,
                    fill_neg=False,
                    rotate_amount=0,
                    rotate_range=(0,360),
                    translate_amount=0,
                    add_pr_noise=0.15,
                    balance=True,
                    normalizers=None,
                    desired_dims=None,
                    antonym_as_neg=True, **kwargs):
        mapinfo = MapInfoDataset()
        for map_name in map_names:
            mapinfo.load_by_name(map_name.strip())
        data_ops = BaseModel.compute_ops(mapinfo, keyword=keyword,
                                         augment_radius=augment_radius,
                                         augment_dfactor=augment_dfactor,
                                         fill_neg=fill_neg,
                                         rotate_amount=rotate_amount,
                                         rotate_range=rotate_range,
                                         translate_amount=translate_amount,
                                         add_pr_noise=add_pr_noise)
        fields = [(FdAbsObjLoc, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdCtxImg, (mapinfo,), {"desired_dims": desired_dims,
                                          "use_nbr": True}),
                  (FdCtxEgoImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdBdgEgoImg, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
                  (FdFoRefAngle, tuple()),
                  (FdLmSym, tuple()),
                  (FdMapName, tuple())]
        print("Building dataset for %s..." % keyword)
        dataset = SpatialRelationDataset.build(keyword, map_names, data_dirpath,
                                               fields=fields,
                                               data_ops=data_ops)
        if normalizers is not None:
            dataset.normalizers = normalizers
        _info_dataset = {keyword: {"fields": fields, "ops": data_ops}}
        return dataset, _info_dataset
