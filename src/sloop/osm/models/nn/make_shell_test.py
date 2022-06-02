# Produce a shell script with bunch of commands to train / test bunch of models
import os
import json
import copy

NN_PATH = os.path.dirname(os.path.abspath(__file__))


cfg = {"data_params:test:rotate_amount": 0}

models = [(2, "random", {"data_params:train:rotate_amount": 0,
                         "data_params:shared:desired_dims": ["28","28"]})]
          # (2, "ego_ctx_foref_angle", {}),
          # (2, "ego_bdg_foref_angle", {}),
          # (2, "ctx_foref_angle",
          #  {"data_params:shared:desired_dims": ["41","41"]})]

# data_dir = os.path.join(NN_PATH, "../../datasets/SL-OSM-Dataset/blind/FOR_only/sg_processed_fixed/")
data_dir = os.path.join(NN_PATH, "results/07192020/test_data/blind")

predicates = [
    ("front", {"data_params:train:rotate_range": [0,360]})
]

cuda = 1

outdir = "experiments"

map_names = {"austin", "cleveland", "denver", "honolulu", "washington_dc"}

f = open("run_train_models.sh", "w")
for kw, cfg_p in predicates:
    print(kw)
    for map_name in map_names:
        print("  " + map_name)
        test_data_path = os.path.join(data_dir, kw, map_name, "test_df.pkl.zip")
        for iteration, model, cfg_m in models:
            print("    %s : %s" % (iteration, model))
            cfg_change = copy.deepcopy(cfg)
            cfg_change.update(cfg_m)
            cfg_change.update(cfg_p)
            if len(cfg_change) > 0:
                config_change = ["--config-change", "\"%s\"" % json.dumps(cfg_change).replace('"', '\\"')]
            command = ["python",
                       "train_model.py",
                       "-dTest", test_data_path,
                       kw,
                       str(iteration),
                       model,
                       os.path.join("config/config-%s.yaml" % map_name),
                       "--log-path", outdir,
                       "--cuda",
                       str(cuda)] + config_change
            f.write(" ".join(command) + "\n")
f.close()
