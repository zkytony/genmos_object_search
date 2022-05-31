# Produce a shell script with bunch of commands to train / test bunch of models
import os
import json
import copy


def pq(s):
    return "\"" + s + "\""

NN_PATH = os.path.dirname(os.path.abspath(__file__))

# To bulk reproduce, change `dd` to point to the root of model results,
# and point data_dir to 
dd = "./results/07212020/models/front"
data_kind = "test"
# data_dir = os.path.join(NN_PATH, "results/07192020/test_data/not_blind_28")

f = open("run_reproduce.sh", "w")
for root, files, dirs in os.walk(dd):
    rootdir = os.path.basename(root)
    if not (rootdir.startswith("iter")\
            and len(rootdir.split("_")) == 3):
        print("Skipped %s" % rootdir)
        continue

    iteration = rootdir.split("_")[0][-1]
    timestamp = rootdir.split("_")[-1]
    baseline = "_".join(rootdir.split("_")[1:-1])
    method = baseline.split(":")[0].replace("-", "_")
    keyword = baseline.split(":")[1]
    map_name = baseline.split(":")[2].replace(",", "_")

    if not os.path.exists(os.path.join(root, "%s_model.pt" % keyword)):
        continue
    
    command = [
        "python",
        "reproduce.py",
        keyword,
        map_name,
        iteration,
        method,
        pq(os.path.join(root, "%s_model.pt" % keyword)),
        pq(os.path.join(root, "%s_df.pkl.zip" % data_kind)),
        data_kind,
        pq(os.path.join(root, "config.yaml")),        
        pq(os.path.join(root, "train_meta.json")),
        "-o",
        pq(os.path.join(root, "reproduce_no_foref"))
    ]
    f.write(" ".join(command) + "\n")
    print("Wrote command: %s" % (" ".join(command)))
f.close()
