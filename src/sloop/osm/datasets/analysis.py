#
import os
import itertools
import numpy as np
import json
from datetime import datetime as dt
from sloop.datasets.SL_OSM_Dataset.mapinfo.map_info_dataset import MapInfoDataset
from sloop.datasets.dataloader import *
from sloop.datasets.utils import *
from sloop.models.nn.plotting import *
from sloop.models.nn.metrics import *
from sloop.models.nn.loss_function import FoRefLoss, clamp_angle

#Note that the datadir below used to be "./SL-OSM-Dataset/blind/FOR_only/sg_processed_fixed"
#they point to the same data.
datadir = "./SL-OSM-Dataset/frame_of_ref/blind/sg_processed"
maps = ["cleveland", "denver", "austin", "honolulu", "washington_dc"]
mapinfo = MapInfoDataset()
for map_name in maps:
    mapinfo.load_by_name(map_name)

desired_dims=(41,41)
fields = [(FdFoRefOrigin, (mapinfo,), {"desired_dims": desired_dims}),
          (FdFoRefAngle, tuple()),
          (FdLmSym, tuple()),
          (FdMapName, tuple())]

predicate = "front"
dataset = SpatialRelationDataset.build(predicate,
                                       maps, datadir,
                                       fields=fields)


lmk_forefs = {}  # map_name -> landmark -> frame of reference
results = {}

for i in range(len(dataset)):
    sample = dataset[i]
    map_name = sample[FdMapName.NAME]
    lmk = sample[FdLmSym.NAME]
    if map_name not in lmk_forefs:
        lmk_forefs[map_name] = {}
    if lmk not in lmk_forefs[map_name]:
        lmk_forefs[map_name][lmk] = []

    foref = read_foref(dataset, [*sample[FdFoRefOrigin.NAME],
                                 sample[FdFoRefAngle.NAME]])
    lmk_forefs[map_name][lmk].append(foref)

for map_name in lmk_forefs:
    results[map_name] = {}
    for lmk in lmk_forefs[map_name]:
        num_forefs = len(lmk_forefs[map_name][lmk])
        pairs = itertools.combinations(np.arange(num_forefs), 2)
        diffs_origin = []
        diffs_angle = []
        for i, j in pairs:
            foref1, foref2 = lmk_forefs[map_name][lmk][i], lmk_forefs[map_name][lmk][j]
            do = euclidean_dist(foref1[:2], foref2[:2])
            diffs_origin.append(do)
            da = math.degrees(clamp_angle(foref1[2] - foref2[2]))
            diffs_angle.append(da)
        results[map_name][lmk] = (diffs_origin, diffs_angle)

# summarize results
# I want to know the FOR average differences for any landmark,
# and for any map.

for map_name in results:
    print("\n%s" % map_name)
    all_diffs_origin = []
    all_diffs_angle = []
    for lmk in results[map_name]:
        diffs_origin, diffs_angle = results[map_name][lmk]
        mean_diffs_origin, ci_diffs_origin = mean_ci_normal(diffs_origin)
        mean_diffs_angle, ci_diffs_angle = mean_ci_normal(diffs_angle)
        print("    %s:" % lmk)
        print("        Mean diff origin: %.4f" % (mean_diffs_origin))
        print("          CI diff origin: %.4f" % (ci_diffs_origin))
        print("        Mean diff angle: %.4f" % (mean_diffs_angle))
        print("          CI diff angle: %.4f" % (ci_diffs_angle))

        all_diffs_origin.extend(diffs_origin)
        all_diffs_angle.extend(diffs_angle)

    mean_all_diffs_origin, ci_all_diffs_origin = mean_ci_normal(all_diffs_origin)
    mean_all_diffs_angle,  ci_all_diffs_angle  = mean_ci_normal(all_diffs_angle)
    print("%s" % map_name)
    print("  | Mean diff origin: %.4f" % (mean_all_diffs_origin))
    print("  |   CI diff origin: %.4f" % (ci_all_diffs_origin))
    print("  | Mean diff angle: %.4f"  % (mean_all_diffs_angle))
    print("  |   CI diff angle: %.4f"  % (ci_all_diffs_angle))

    # Actually save these in the same format as other baselines
    start_time = dt.now()
    timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    result_dir = os.path.join("analysis", "iter2_annotator:%s:%s_%s"
                              % (predicate, map_name, timestr),
                              "metrics", "test")
    os.makedirs(result_dir, exist_ok=True)
    result = {
        "__summary__": {
            "true_pred_angle_diff": {
                "ci-95": ci_all_diffs_angle,
                "mean": mean_all_diffs_angle
            },
            "true_pred_origin_diff": {
                "ci-95": ci_all_diffs_origin,
                "mean": mean_all_diffs_origin
            }
        },
        "true_pred_angle_diff": all_diffs_angle,
        "true_pred_angle_origin": all_diffs_origin
    }
    with open(os.path.join(result_dir, "foref_deviations.json"), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
