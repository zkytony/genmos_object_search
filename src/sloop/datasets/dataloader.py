# This model should learn the distribution
#
#  Pr ( spatial_relation = True | FoR, point, landmark_features )
#
# for a given spatial_relation.
import torch
import os
import numpy as np
import pickle
import copy
import sys, inspect
import pandas as pd
import math
import random
from torch.utils.data import Dataset
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.utils import json_safe
from sloop.datasets.utils import *

#### FIELDS ####
class Field:
    """a kind of information. E.g. object location, frame of reference etc."""
    NAME = "unknown"
    NORMALIZE = True  # Whether this field should be normalized
    USE_NORM_FUNC="use_norm_func"  # just a constant. not to be inherited and changed

    @classmethod
    def get_val(cls, sample, *args, **kwargs):
        raise NotImplementedErrorp

class FdObjLoc(Field):
    NAME = "relative_obj_xy"

    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        """Returns object location in the map with
        size given by `desireddim`. If `desireddim` is None,
        then the map size comes from the original map
        size stored in mapinfo for this map."""
        original_dims = mapinfo.map_dims(sample["map_name"])
        if desired_dims is None:
            desired_dims = mapinfo.map_dims(sample["map_name"])

        landmark_symbol = sample["landmark_symbol"]
        if mapinfo.map_name_of(landmark_symbol) is None:
            # This is not a landmark symbol
            return None
        landmark_center = mapinfo.center_of_mass(landmark_symbol, sample["map_name"])
        rel_obj_loc = np.array([sample["obj_loc"][0] - landmark_center[0],
                                sample["obj_loc"][1] - landmark_center[1]])
        w,l = original_dims
        rel_obj_loc = rotate_pt([w/2, l/2], rel_obj_loc, sample.get("angle_diff", 0.0))
        if desired_dims != original_dims:
            rel_obj_loc = np.array(scale_point(rel_obj_loc,
                                               original_dims, desired_dims))
        return rel_obj_loc

class FdAbsObjLoc(Field):
    NAME = "absolute_obj_xy"

    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        abs_obj_loc = np.array(sample["obj_loc"])
        if mapinfo is not None and desired_dims is not None:
            original_dims = mapinfo.map_dims(sample["map_name"])
            if original_dims != desired_dims:
                abs_obj_loc = np.array(scale_point(abs_obj_loc,
                                                   original_dims, desired_dims))
        return abs_obj_loc

class FdAbsLmkLoc(Field):
    NAME = "absolute_landmark_xy"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        original_dims = mapinfo.map_dims(sample["map_name"])
        if desired_dims is None:
            desired_dims = mapinfo.map_dims(sample["map_name"])

        landmark_symbol = sample["landmark_symbol"]
        if mapinfo.map_name_of(landmark_symbol) is None:
            # This is not a landmark symbol
            return None
        w,l = original_dims
        landmark_center = np.array(mapinfo.center_of_mass(landmark_symbol, sample["map_name"]))
        landmark_center = rotate_pt([w/2, l/2], landmark_center, sample.get("angle_diff", 0.0))
        landmark_center = translate_point(landmark_center, sample.get("trans_vec", (0, 0)))
        if desired_dims != original_dims:
            landmark_center = np.array(scale_point(landmark_center,
                                                   original_dims, desired_dims))
        return landmark_center

class FdObjLocPolar(Field):
    NAME = "relative_obj_xy_polar"
    @classmethod
    def get_val(cls, sample, mapinfo, **kwargs):
        rel_obj_loc = to_polar(FdObjLoc.get_val(sample, mapinfo, **kwargs))
        return rel_obj_loc

class FdAbsObjLocPolar(Field):
    NAME = "relative_obj_xy_polar"
    @classmethod
    def get_val(cls, sample, mapinfo, **kwargs):
        rel_obj_loc = to_polar(FdAbsObjLoc.get_val(sample, mapinfo, **kwargs))
        return rel_obj_loc

class FdAbsLmkLocPolar(Field):
    NAME = "relative_obj_xy_polar"
    @classmethod
    def get_val(cls, sample, mapinfo, **kwargs):
        rel_obj_loc = to_polar(FdAbsLmkLoc.get_val(sample, mapinfo, **kwargs))
        return rel_obj_loc

class FdFoRefOrigin(Field):
    NAME = "frame_of_ref_origin"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        foref_origin = np.array([sample["frame_of_refs"][0][0],
                                 sample["frame_of_refs"][0][1]])
        if mapinfo is not None and desired_dims is not None:
            original_dims = mapinfo.map_dims(sample["map_name"])
            if original_dims != desired_dims:
                foref_origin = np.array(scale_point(foref_origin,
                                                    original_dims, desired_dims))
        return foref_origin

class FdFoRefAngle(Field):
    NAME = "frame_of_ref_angle"
    @classmethod
    def get_val(cls, sample):
        return np.array([sample["frame_of_refs"][1]])

class FdBdgImg(Field):
    NAME = "bdg_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None, border=False):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None
        arr = make_img(mapinfo, sample["map_name"], [sample["landmark_symbol"]],
                       landmark_border=border)
        arr = rotate_img(arr, sample.get("angle_diff", 0.0))
        arr = translate_img(arr, sample.get("trans_vec", (0, 0)))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()

# Deprecated.
class FdBdgEgoImgDeprecated(Field):
    NAME = "ego_bdg_img_deprecated"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None, border=False):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None
        arr = make_img(mapinfo, sample["map_name"], [sample["landmark_symbol"]],
                       landmark_border=border)
        arr = to_ego_img(arr, sample["map_name"], sample["landmark_symbol"], mapinfo)
        arr = rotate_img(arr, sample.get("angle_diff", 0.0))
        arr = translate_img(arr, sample.get("trans_vec", (0, 0)))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()

class FdMapImg(Field):
    NAME = "map_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None

        arr = make_img(mapinfo, sample["map_name"],
                       mapinfo.landmarks_for(sample["map_name"]))
        arr = rotate_img(arr, sample.get("angle_diff", 0.0))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()

class FdMapEgoImg(Field):
    NAME = "ego_map_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None
        arr = make_img(mapinfo, sample["map_name"],
                       mapinfo.landmarks_for(sample["map_name"]))
        arr = to_ego_img(arr, sample["map_name"], sample["landmark_symbol"], mapinfo)
        arr = rotate_img(arr, sample.get("angle_diff", 0))
        arr = translate_img(arr, sample.get("trans_vec", (0, 0)))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()

class FdCtxImg(Field):
    NAME = "ctx_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None, use_nbr=False):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None
        if use_nbr:
            arr = make_nbr_img(mapinfo, sample["map_name"],
                               sample["landmark_symbol"],
                               landmark_border=False)
        else:
            arr = make_context_img(mapinfo, sample["map_name"],
                                   sample["landmark_symbol"], dist_factor=2.0,
                                   landmark_border=False)
        arr = rotate_img(arr, sample.get("angle_diff", 0.0))
        arr = translate_img(arr, sample.get("trans_vec", (0, 0)))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()

class FdCtxEgoImg(Field):
    NAME = "ctx_ego_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None, mapsize=(28,28),
                bdg_only=False):
        if mapinfo.map_name_of(sample["landmark_symbol"]) is None:
            return None
        arr = ego_lmk_map(sample["landmark_symbol"],
                          sample["map_name"], mapinfo, mapsize=mapsize,
                          landmark_border=False, use_context=(not bdg_only))
        arr = rotate_img(arr, sample.get("angle_diff", 0.0))
        arr = translate_img(arr, sample.get("trans_vec", (0, 0)))
        if desired_dims is not None:
            arr = scale_map(arr, desired_dims)
        return arr.flatten()


class FdBdgEgoImg(Field):
    NAME = "bdg_ego_img"
    @classmethod
    def get_val(cls, sample, mapinfo, desired_dims=None, mapsize=(28,28)):
        return FdCtxEgoImg.get_val(sample, mapinfo,
                                   desired_dims=desired_dims, mapsize=mapsize,
                                   bdg_only=True)


class FdProbSR(Field):
    NAME =  "prob_sr"

    @classmethod
    def get_val(cls, sample):
        if "confidence" not in sample:
            return np.array([1.0])
        else:
            return np.array([sample["confidence"]])

class FdLmSym(Field):
    """Landmark symbol"""
    NAME = "landmark_symbol"
    NORMALIZE = False
    @classmethod
    def get_val(cls, sample):
        return sample["landmark_symbol"]

class FdMapName(Field):
    """Landmark symbol"""
    NAME = "map_name"
    NORMALIZE = False
    @classmethod
    def get_val(cls, sample):
        return sample["map_name"]

class FdHint(Field):
    """Landmark symbol"""
    NAME = "hint"
    NORMALIZE = False
    @classmethod
    def get_val(cls, sample):
        return sample["hint"]

NAME_TO_FIELD = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj):
        if issubclass(obj, Field):
            NAME_TO_FIELD[obj.NAME] = obj
########## NO MORE FIELDS ############

#### Data Operators ####
class DataOperator:
    """Used to apply transformations to the given data samples (list).
    For example: Add negative samples, augment positive samples etc."""
    NAME = "unknown"
    @classmethod
    def apply(cls, data, *args, **kwargs):
        """Apply this operation on the given data."""
        pass


class OpAugPositive(DataOperator):
    """Augment positive samples by expanding"""
    NAME = "augment_pos"
    @classmethod
    def apply(cls, data, mapinfo, radius, dfactor=None, verbose=False, keep_highest=True):
        """
        dfactor (float): Downgrade factor. If not none, then will downgrade the
            confidence in the spatial relation probability based on the
            distance of the cell to the center (i.e. obj_loc) multiplied
            by this factor.

            When this is provided, in order to avoid one location being
            assigned two different probability values, we will keep the
            sample(s) with highest probability value when such case occurs.
        """
        print("  Augmenting positive examples by expanding a radius on existing ones.")
        if len(data) > 0:
            _keyword = data[0]["spatial_keyword"]
        else:
            return data

        new_count = 0
        new_data = []
        loc_to_prob = {(x,y):[]
                       for x in range(mapinfo.map_dims[0])
                       for y in range(mapinfo.map_dims[1])}
        for sample in data:
            assert sample["spatial_keyword"] == _keyword,\
                "spatial keyword mismatch (%s != %s). Unexpected!!"\
                % (keyword, sample["spatial_keyword"])

            obj_loc = sample["obj_loc"]
            loc_to_prob[obj_loc].append((1.0, sample))

            # Get cells within radius centered at obj_loc
            cells_in_range = set()
            width, length = mapinfo.map_dims
            for x in range(width):
                for y in range(length):
                    if (x,y) != obj_loc:
                        if euclidean_dist((x,y), obj_loc) <= radius:
                            cells_in_range.add((x,y))

            for x,y in cells_in_range:
                if dfactor is None:
                    new_sample = copy.deepcopy(sample)
                    new_sample["obj_loc"] = (x,y)
                    new_sample["confidence"] = 1.0
                else:
                    prob = dfactor**(int(round(euclidean_dist((x,y), obj_loc))))
                    new_sample = copy.deepcopy(sample)
                    new_sample["obj_loc"] = (x,y)
                    new_sample["confidence"] = prob
                loc_to_prob[(x,y)].append((new_sample["confidence"], new_sample))

        # keep samples with highest confidence
        for x,y in loc_to_prob:
            if len(loc_to_prob[(x,y)]) > 0:
                if keep_highest:
                    max_confidence = max(loc_to_prob[(x,y)], key=lambda t: t[0])[0]
                    for conf, sample in loc_to_prob[(x,y)]:
                        if abs(conf - max_confidence) < 1e-6:
                            new_data.append(sample)
                else:
                    for _, sample in loc_to_prob[(x,y)]:
                        new_data.append(sample)
        if verbose:
            print("New samples count: %d" % (len(new_data) - len(data)))
        return new_data

class OpFillNeg(DataOperator):
    NAME = "augment_neg"
    @classmethod
    def apply(cls, data, mapinfo, keyword, verbose=False, no_duplicate_loc=True):
        """If a grid cell does not have positive data, then it will be treated
        as negative.
        `no_duplicate_loc` is true if you only want to create one negative sample
        per grid cell in the map."""
        print("  Adding negative examples.")
        original_len = len(data)
        new_data = data
        positive_locs = {}  # map name to landmark to set
        for sample in data:
            map_name = sample["map_name"]
            lmsym = sample["landmark_symbol"]
            if map_name not in positive_locs:
                positive_locs[map_name] = {}
            if mapinfo.map_name_of(lmsym) is None:
                # This landmark symbol does not belong to the map
                continue
            if sample["landmark_symbol"] not in positive_locs[map_name]:
                positive_locs[map_name][sample["landmark_symbol"]] = set()
            positive_locs[map_name][sample["landmark_symbol"]].add(sample["obj_loc"])

        negative_recorded = set()
        for map_name in positive_locs:
            w, l = mapinfo.map_dims(map_name)
            for lmsym in positive_locs[map_name]:
                all_locs = set({(x,y)
                                for x in range(w)
                                for y in range(l)})
                negative_locs = all_locs - positive_locs[map_name][lmsym]
                for loc in negative_locs:
                    if no_duplicate_loc and loc in negative_recorded:
                        continue
                    dummy_sample = {
                        "map_name": map_name,
                        "obj_num": '1',
                        "obj_exist": "bike",
                        "robot_loc": "441",
                        "obj_loc": loc,
                        "spatial_keyword": keyword,
                        "landmark_symbol": lmsym,
                        "landmarks_mentioned": {lmsym},
                        "hint": "NO HINT - Sample Added During Data Processing.",
                        "relations": [],
                        "confidence": 1e-9
                    }
                    negative_recorded.add(loc)
                    new_data.append(dummy_sample)
        if verbose:
            print("New samples count: %d" % (len(new_data) - original_len))
        return new_data

class OpFlipProb(DataOperator):
    NAME = "flip_prob"
    @classmethod
    def apply(cls, data, verbose=True):
        """Assume all samples from data are positive samples"""
        keyword = data[0]["spatial_keyword"]
        print("Flipping the probability of samples from %s." % keyword)
        new_data = []
        for sample in data:
            new_sample = copy.deepcopy(sample)
            if "confidence" in sample:
                new_sample["confidence"] = 1.0 - sample["confidence"]
            else:
                new_sample["confidence"] = 1e-9
            new_data.append(new_sample)
        if verbose:
            print("Examples added through FLIPPING count: %d" % len(new_data))
        return new_data

class OpProbAddNoise(DataOperator):
    NAME = "add_noise"
    @classmethod
    def apply(cls, data, noise=0.2, verbose=True):
        """Assume all samples from data are positive samples"""
        print("  Adding noise (+/-%.3f) to the confidence" % noise)
        new_data = []
        for sample in data:
            new_sample = copy.deepcopy(sample)
            if "confidence" not in sample:
                new_sample["confidence"] = 1.0 + random.uniform(-noise, noise)
            else:
                new_sample["confidence"] += random.uniform(-noise, noise)
            new_sample["confidence"] = min(max(new_sample["confidence"], 0.0), 1.0)
            new_data.append(new_sample)
        return new_data

class OpRandomRotate(DataOperator):
    NAME = "rotate"
    @classmethod
    def apply(cls, data, mapinfo, amount=200, rotate_range=(0, 360)):
        """Rotate frame of reference randomly;
        Rotation center should be the center of the map (21x21)"""
        print("  Augmentation by randomly rotate each example %d times" % amount)
        new_data = []
        for k, sample in enumerate(data):
            if sample["map_name"] not in mapinfo.landmarks:
                continue  # this map was not loaded.
            foref_angle = sample["frame_of_refs"][1]
            for i in range(amount):
                new_sample = copy.deepcopy(sample)
                angle_diff = random.uniform(math.radians(rotate_range[0]),
                                            math.radians(rotate_range[1]))
                new_angle = foref_angle + angle_diff

                # Rotate frame of reference w.r.t. map center
                w,l = mapinfo.map_dims(sample["map_name"])
                new_origin = rotate_pt([w/2, l/2], sample["frame_of_refs"][0], angle_diff)
                new_sample["frame_of_refs"] =\
                    [new_origin, new_angle]
                new_sample["angle_diff"] = angle_diff

                # Rotate object location w.r.t. map center
                new_obj_loc = rotate_pt([w/2, l/2], sample["obj_loc"], angle_diff)
                new_sample["obj_loc"] = new_obj_loc
                new_data.append(new_sample)
                sys.stdout.write("  ....... [%d/%d]\t(%d/%d)\r" % (i+1, amount, k+1, len(data)))
            new_data.append(sample)
        # print("Extra data: %d" % (len(new_data) - len(data)))
        return new_data


class OpRandomTranslate(DataOperator):
    NAME = "translate"
    @classmethod
    def apply(cls, data, mapinfo, amount=200, radius_ratio=0.15):
        """Translate the frame of reference randomly within a radius."""
        print("  Augmentation by randomly translate %d times" % amount)
        new_data = []
        for k, sample in enumerate(data):
            if sample["map_name"] not in mapinfo.landmarks:
                continue  # this map was not loaded.
            foref_origin = sample["frame_of_refs"][0]
            w, l = mapinfo.map_dims(sample["map_name"])
            trans_size = w * radius_ratio
            for i in range(amount):
                new_sample = copy.deepcopy(sample)
                trans_x = int(round(random.uniform(-trans_size, trans_size)))
                trans_y = int(round(random.uniform(-trans_size, trans_size)))
                new_origin = translate_point(foref_origin, (trans_x, trans_y))
                if 0 <= new_origin[0] < w\
                   and 0 <= new_origin[1] < l:
                    # Translate the frame of reference
                    new_sample["frame_of_refs"] = [new_origin,
                                                   new_sample["frame_of_refs"][1]]

                    # Translate the object location
                    new_obj_loc = translate_point(sample["obj_loc"], (trans_x, trans_y))
                    new_sample["obj_loc"] = new_obj_loc
                    new_sample["trans_vec"] = (trans_x, trans_y)
                    new_data.append(new_sample)
                sys.stdout.write("  ....... [%d/%d]\t(%d/%d)\r" % (i+1, amount, k+1, len(data)))
            new_data.append(sample)
        # print("Extra data: %d" % (len(new_data) - len(data)))
        return new_data


class OpBalance(DataOperator):
    """This operation should be performed at last."""
    NAME = "balance"
    @classmethod
    def apply(cls, data, amount_to_equal):
        """Add or drop samples so that the number of resulting examples
        is equal to `amount_to_equal`."""
        assert type(data) == list, "Data should be a list. But it is a %s" % type(data)
        amount_now = len(data)
        new_data = copy.deepcopy(data)
        if amount_now < amount_to_equal:
            print("  Balancing data to reach %d samples. "
                  "Will add %d samples randomly" % (amount_to_equal, amount_to_equal - amount_now))
            # Randomly sample own data
            for i in range(amount_to_equal - amount_now):
                random_sample = copy.deepcopy(random.sample(data, 1)[0])
                new_data.append(random_sample)
        else:
            print("  Balancing data to reach %d samples. "
                  "Will remove % samples randomly" % (amount_to_equal, amount_now - amount_to_equal))
            # randomly remove own data
            for i in range(amount_now - amount_to_equal):
                idx = random.randrange(0, len(new_data))
                del new_data[idx]
        assert len(new_data) == amount_to_equal
        return new_data


class SpatialRelationDataset(Dataset):
    """Dataset that
    (spatial_relation, frame of reference, point, landmark_features)
    """
    @classmethod
    def build(cls, keyword, map_names, dirpath,
              fields=[], data_ops=[], normalizers=None):
                 # fields=[FdObjLoc, FdProbSR],
                 # data_op=[(OpAugmentPositive)]):
        """
        map_names (list) List of map names to build the dataset from
        dirpath (str) path to directory that contains the data per predicate.
             Each predicate should have a corresponindg .pickle file.
        fields (list): List of (Field, args) that will be the _kinds_ of info the dataset
             contains per row (e.g. object location, relation probability).
        data_ops (list): Operations to apply to the data right after it's loaded.
             For example, augment positive samples. List of (DataOperator, [args])
             ORDER MATTERS.
        normalizers (dict): Mapping from field name to (min, max) values. If None,
             then a normalizer dictionary will be built for this dataset.
        """
        with open(os.path.join(dirpath, "%s-positive.pkl" % keyword), "rb") as f:
            data = pickle.load(f)

        for item in data_ops:
            if len(item) == 2:
                op, op_args, op_kwargs = item[0], item[1], {}
            else:
                op, op_args, op_kwargs = item
            print("[Applying operator :%s:]" % op.NAME)
            data = op.apply(data, *op_args, **op_kwargs)

        # Process the data. Get the value for each field.
        data_by_field = {f[0].NAME: [] for f in fields}
        # map from field name to a (min, max) of that field. (min-max normalization)
        for i, sample in enumerate(data):
            sys.stdout.write("Building dataframe ... [%d/%d]\r" % (i+1, len(data)))
            if sample["map_name"] not in map_names:
                continue
            bad_sample = False
            fd_vals = {}
            for item in fields:
                if len(item) == 2:
                    fd, fd_args, fd_kwargs = item[0], item[1], {}
                else:
                    fd, fd_args, fd_kwargs = item
                val = fd.get_val(sample, *fd_args, **fd_kwargs)
                if val is None:
                    bad_sample = True
                    break
                fd_vals[fd.NAME] = val
            if bad_sample:
                continue

            for name in fd_vals:
                data_by_field[name].append(fd_vals[name])

        return SpatialRelationDataset(pd.DataFrame(data_by_field),
                                      map_names, normalizers=normalizers)

    def __init__(self, df, map_names, normalizers=None):
        self.df = df
        self.map_names = map_names
        if normalizers is None:
            normalizers = {}
            for field_name, field_data in df.iteritems():
                if not NAME_TO_FIELD[field_name].NORMALIZE:
                    continue
                normalizers[field_name] = (float("inf"), float("-inf"))
                for i, val in field_data.iteritems():
                    minval, maxval = normalizers[field_name]
                    minval = min(np.min(val), minval)
                    maxval = max(np.max(val), maxval)
                    normalizers[field_name] = (float(minval), float(maxval))
        self.normalizers = normalizers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Always normalize whenever you get item"""
        dct = self.df.iloc[idx].to_dict()
        for key in dct:
            if key in self.normalizers:
                val = self.normalize(key, dct[key])
                dct[key] = torch.from_numpy(val).float()
        return dct

    def get(self, idx, normalize=False):
        if normalize:
            return self[idx]
        else:
            return self.df.iloc[idx].to_dict()

    def normalize(self, field_name, value):
        if self.normalizers[field_name] == Field.USE_NORM_FUNC:
            return NAME_TO_FIELD[field_name].normalize(value)
        else:
            minval, maxval = self.normalizers[field_name]
            return (value - minval) / (maxval - minval)

    def rescale(self, field_name, value):
        if self.normalizers[field_name] == Field.USE_NORM_FUNC:
            return NAME_TO_FIELD[field_name].rescale(value)
        else:
            minval, maxval = self.normalizers[field_name]
            return value * (maxval - minval) + minval

    def input_size(self, model):
        s0 = self.df.iloc[0]
        totlen = 0
        for f in model.input_fields:
            if hasattr(s0, '__len__'):
                totlen += len(s0[f])
            else:
                totlen += 1
        return totlen

    def split(self, ratio, shuffle=True):
        """Split the data into two parts. The normalizers will be KEPT THE SAME"""
        df = self.df
        if shuffle:
            df = self.df.sample(frac=1)
        total_len = len(df)
        indices = df.index[:int(round(total_len * ratio))]

        ratio_df = df.iloc[indices]
        remain_df = df.drop(indices)  # FOR SOME REASON: The resulting `remain_df`
                                      # has varying number of rows when this method is called.
        return SpatialRelationDataset(remain_df, self.map_names, self.normalizers),\
            SpatialRelationDataset(ratio_df, self.map_names, self.normalizers)

    def append(self, dataset):
        """Add in the new dataset. The normalizers will be UPDATED."""
        return SpatialRelationDataset(self.df.append(dataset.df),
                                      self.map_names, normalizers=None)


def test():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import seaborn
    dirpath = os.path.join("amt", "sg_processed")

    map_names = ["dorrance", "faunce"]

    mapinfo = MapInfoDataset()
    for map_name in map_names:
        mapinfo.load_by_name(map_name)

    dataset = SpatialRelationDataset.build("near", map_names, dirpath,
                                           fields=[(FdObjLoc, (mapinfo,)),
                                                   (FdProbSR, tuple())],
                                           data_ops=[(OpAugPositive, (mapinfo, 0), {"dfactor": 0.7})])
    dataloader = DataLoader(dataset=dataset, batch_size=15, shuffle=True)
    for batch in dataloader:
        print(batch)

    # plot the samples
    xvals, yvals, confs = [], [], []
    for index, row in dataset.df.iterrows():
        xvals.append(row[FdObjLoc.NAME][0])
        yvals.append(row[FdObjLoc.NAME][1])
        confs.append(row[FdProbSR.NAME][0])

    df = pd.DataFrame({"x": xvals,
                       "y": yvals,
                       "confs": confs})
    seaborn.scatterplot(x="x", y="y", hue="confs",
                        marker="s", data=df)
    plt.show()


if __name__ == "__main__":
    test()
