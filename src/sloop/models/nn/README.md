# How to train models

Use the script `train_model.py` as follows
```
usage: train_model.py [-h] [-dRaw DATADIR] [-dTrain TRAIN_DATA]
                      [-dVal VAL_DATA] [-dTest TEST_DATA]
                      [--log-path LOG_PATH] [--cuda CUDA]
                      [--config-change CONFIG_CHANGE]
                      keyword iteration model_type config_file
```

Example:
```
python train_model.py -dRaw ../../datasets/SL-OSM-Dataset/frame_of_ref/blind/sg_processed/ front 2 ego_ctx_foref_angle config/config-austin.yaml
```

The `datadir` is a path to the directory that contains the `<keyword>-positive.pkl` files.




# How to reproduce a result

Once you done training, the `train_model.py` script will evaluate the trained model's performance. If you want to reproduce those performance results, you can go to the model's log directory, and use the command saved in `reproduce_train.txt` or `reproduce_val.txt` etc. to reproduce those results. This command will also be printed on the terminal during the running of `train_model.py`


# How to build a model

You can build any model as long as they have a `get_data` function, `Train` function, and `Plot` function similar to the ones in `pr_to_fref_model.py`, **and `Eval` function (new!)**. They can take parameters from the `config` yaml file. These models should be placed under a directory named `iter#` for organization.

Once you created a model, you should edit the `models/nn/__init__.py` file,
import your model, and append it to the `MODELS_BY_ITER` dictionary with the appropriate iteration number as the key.


# Note

The models we ended up using are in `iter2`.
