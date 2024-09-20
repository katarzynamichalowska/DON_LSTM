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
import argparse
sys.path.insert(0, './modules')
sys.path.insert(0, './modules/fno')
from modules.plotting import plot_history
from modules.data_manipulation import resample_g_xt
from modules.fno.data_manipulation import preprocess_data
from modules.fno.utilities3 import UnitGaussianNormalizer, LpLoss, count_params
from modules.fno.fno_operations import FNO2d, FNO3d, FNO2d_RNN, FNO2d_LSTM, FNO2d_GRU, FNO3d_RNN, SpectralConv2d, data_loaders, train_step, evaluate_model, train_model
from modules.fno.Adam import Adam
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions

import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
import params_fno as params
import torch.nn as nn

# Seed
#torch.manual_seed(0)
#np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "8" 

time_first=True

# Load model and continue training
MODEL_TO_LOAD = None

#--------------------     PARAMS FOR TESTING     --------------------#

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', required=False, action="store_true",
                    help="Run only 2 epochs and print the losses; save to a test folder.")
parser.add_argument('-m', '--model', required=False,
                    help="Read a model and continue training from the last epoch.")
args = parser.parse_args()
in_testing, model = args.test, args.model

if in_testing:
    N_EPOCHS = 3
    LOG_OUTPUT_FREQ = 1
    CHECKPOINTS_FREQ = 2
else:
    N_EPOCHS = params.N_EPOCHS
    LOG_OUTPUT_FREQ = params.LOG_OUTPUT_FREQ
    CHECKPOINTS_FREQ = params.CHECKPOINTS_FREQ


#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(params, is_testing=in_testing)

# Directories
checkpoint_dir = os.path.join(output_folder, "checkpoints")
logs_path = os.path.join(output_folder, "log.out")
params_log_path = os.path.join(output_folder, "params.txt")
training_log_path = os.path.join(output_folder, "training_log.out")


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(params_log_path, 'w') as params_log:
    params_log.write(log_functions.print_params(params))
params_log.close()

logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))
logs.write(log_functions.print_params(params))

PATH_MODEL = os.path.join(output_folder, "fourier2d")
CHECKPOINT_PATH = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
CUDA = True
t1 = default_timer()

#--------------------     LOAD DATA     --------------------#

data = np.load(params.DATAFILE)
u, xt, g_u = data['u'], data['xt'], data['g_u']

#-------------------- DATA PREPROCESSING --------------------#


if params.RESAMPLE_I is not None:
    g_u, xt = resample_g_xt(g_u, xt, params.RESAMPLE_I)
T_IN = 1
S = u.shape[1]
t_len = int(g_u.shape[1]/S)

if params.RESAMPLE_I is not None:
    T = int(t_len/params.RESAMPLE_I)
else:
    T = t_len

data_p = preprocess_data(u, g_u, S, T, T_IN, 
                         train_idx=params.TRAIN_IDX, val_idx=params.VAL_IDX, test_idx=params.TEST_IDX,
                         train_perc=params.TRAIN_PERC, val_perc=params.VAL_PERC, 
                         nr_timestamps=params.NR_TIMESTAMPS, input_3d=params.INPUT_3D)


#u_normalizer, g_u_normalizer = data_p['u_normalizer'], data_p['g_u_normalizer']

# Data loaders
train_loader, val_loader, test_loader = data_loaders(data=data_p, batch_size=params.BATCH_SIZE, normalized=False, shuffle_train=False)

t2 = default_timer()
print('Preprocessing finished, time used:', t2-t1)

#--------------------       MODEL       --------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_TO_LOAD is None:
    if params.RNN_LAYER is None:
        if params.INPUT_3D:
            model = FNO3d(params.MODES, params.MODES, params.MODES, params.WIDTH)
        else:
            model = FNO2d(params.MODES, params.MODES, params.WIDTH)
    else:
        if params.INPUT_3D:
            model = FNO3d_RNN(params.MODES, params.MODES, params.MODES, params.WIDTH, t_len, S)
        else:
            if params.LSTM:
                model = FNO2d_LSTM(params.MODES, params.MODES, params.WIDTH, t_len, S)
            elif params.GRU:
                model = FNO2d_GRU(params.MODES, params.MODES, params.WIDTH, t_len, S)
            else:
                model = FNO2d_RNN(params.MODES, params.MODES, params.WIDTH, t_len, S)
else:
    model = torch.load(MODEL_TO_LOAD)

if CUDA:
    #model = nn.DataParallel(model) 

    model = model.cuda()
    #g_u_normalizer.cuda()
else:
    model = model.float()


if params.INPUT_3D:
    input_size = data_p['u_train'].shape[1:]
else:
    input_size = data_p['u_train'].shape[1:]
#print(summary(model, input_size))

#logs.write(print(summary(model, data_p['u_train'].shape[1:])))

print("Nr model parameters: ", count_params(model))

logs.close()
logs = open(logs_path, 'a')

#--------------------       TRAIN       --------------------#

optimizer = Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=1e-4)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.STEP_SIZE, gamma=params.GAMMA)
#myloss = LpLoss(size_average=False)


history, log_history = train_model(model, 
                                    train_loader=train_loader, 
                                    val_loader=val_loader, 
                                    n_epochs=N_EPOCHS, 
                                    optimizer=optimizer, 
                                    scheduler=scheduler, 
                                    checkpoints_folder=checkpoint_dir,
                                    checkpoints_freq=CHECKPOINTS_FREQ, 
                                    log_output_freq=LOG_OUTPUT_FREQ, 
                                    log_temp_path=training_log_path, 
                                    last_cp=0, cuda=True)

torch.save(model, PATH_MODEL)


plot_history(history, logy=False, savedir=os.path.join(output_folder, "training_history.png"))
plot_history(history, logy=True, savedir=os.path.join(output_folder, "training_history_log.png"))
plot_history(history, logy=False, plot_train=False, savedir=os.path.join(output_folder, "training_history_val.png"))
plot_history(history, logy=True, plot_train=False, savedir=os.path.join(output_folder, "training_history_val_log.png"))

logs.write(log_functions.print_training_history(log_history))


mse_test, _, _ = evaluate_model(model, data_loader=test_loader, inverse_norm=False, g_u_normalizer=None, cuda=CUDA)

logs.write("RMSE test (normal scale): {}\n".format(np.round(np.sqrt(mse_test),5)))

logs.write(log_functions.print_time("Program finished"))
logs.close()