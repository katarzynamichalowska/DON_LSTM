"""
To test, run: 
python train_fno.py -t

This file is used for training vanilla FNOs and FNO-RNN models.

To generate training data, first run data_generation_kdv.py or specify a training file in params_fno.py (DATAFILE)
Parameters for of the trained models are available in the folder "./models" in params.txt files of specific models.
The FNO and FNO-RNN models are named: "fno_vanilla*", "fno_rnn*". 
Training parameters can be specified in the params_fno.py file.
"""

import sys
import os
sys.path.insert(0, './modules')
sys.path.insert(0, './modules/fno')
from modules.plotting import plot_full_history
from modules.data_manipulation import resample_g_xt
from modules.fno.data_manipulation import preprocess_data
from modules.fno.utilities3 import UnitGaussianNormalizer, LpLoss, count_params
from modules.fno.fno_operations import data_loaders, evaluate_model
from modules.fno.Adam import Adam
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions
from modules.load_model_torch import load_fno_model

import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
import torch.nn as nn

# Seed
#torch.manual_seed(0)
#np.random.seed(0)

#--------------------     PARAMS FOR TESTING     --------------------#

p = dir_functions.load_params('params_fno_test.yml')

#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(p["MODEL_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["CHECKPOINT_TO_TEST"])

test_folder = dir_functions.make_output_dir(p["TEST_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["CHECKPOINT_TO_TEST"])
test_folder = os.path.join(test_folder, f"checkpoint_{p['CHECKPOINT_TO_TEST']}")
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
# Directories
checkpoint_dir = os.path.join(output_folder, "checkpoints")
training_log_path = os.path.join(output_folder, "training_log.out")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

PATH_MODEL = os.path.join(output_folder, p["MODELNAME"])
t1 = default_timer()

#--------------------     LOAD DATA     --------------------#

data = np.load(p["DATAFILE"])
u, xt, g_u = data['u'], data['xt'], data['g_u']

#-------------------- DATA PREPROCESSING --------------------#


if p["RESAMPLE_I"] is not None:
    g_u, xt = resample_g_xt(g_u, xt, p["RESAMPLE_I"])
T_IN = 1
S = u.shape[1]
t_len = int(g_u.shape[1]/S)

if p["RESAMPLE_I"] is not None:
    T = int(t_len/p["RESAMPLE_I"])
else:
    T = t_len

data_p = preprocess_data(u, g_u, S, T, T_IN, 
                         train_idx=p["TRAIN_IDX"], val_idx=p["VAL_IDX"], test_idx=p["TEST_IDX"],
                         train_perc=p["TRAIN_PERC"], val_perc=p["VAL_PERC"], 
                         nr_timestamps=p["NR_TIMESTAMPS"], input_3d=p["INPUT_3D"])

# Data loaders
train_loader, val_loader, test_loader = data_loaders(data=data_p, batch_size=p["BATCH_SIZE"], normalized=False, shuffle_train=False)

t2 = default_timer()
print('Preprocessing finished, time used:', t2-t1)

#--------------------       MODEL       --------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_fno_model(output_folder, p["MODELNAME"], device, cp=p["CHECKPOINT_TO_TEST"])

if device == torch.device("cuda"):
    model = model.cuda()
    #g_u_normalizer.cuda()
else:
    model = model.float()


if p["INPUT_3D"]:
    input_size = data_p['u_train'].shape[1:]
else:
    input_size = data_p['u_train'].shape[1:]
#print(summary(model, input_size))

#logs.write(print(summary(model, data_p['u_train'].shape[1:])))

print("Nr model parameters: ", count_params(model))

#-------------------- MODEL EVALUATION --------------------#

# TODO rescale back to original scale
mse_test, y_pred_all, y_all = evaluate_model(model, data_loader=test_loader, inverse_norm=False, g_u_normalizer=None, cuda=p["CUDA"])

with open(os.path.join(test_folder, "stats.txt"), 'w') as f:
    f.write("RMSE test (normal scale): {}\n".format(np.round(np.sqrt(mse_test),5)))

plot_full_history(test_folder, os.path.join(output_folder, "training_log.out"), plot_name="full_training_history", checkpoint=p['CHECKPOINT_TO_TEST'])