import sys
sys.path.insert(0, './')
import modules.animation as a
import numpy as np
import os
from modules.evaluate import evaluate_data
from modules.training import eval_in_batches
from modules.model_definition import compile_model
from modules.data_manipulation import preprocess_data
from modules.load_model import load_model
from modules.plotting import plot_full_history
import modules.dir_functions as dir_functions

p = dir_functions.load_params(os.path.join('deeponets', 'params_test.yml'))

#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(p["MODEL_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["CHECKPOINT_TO_TEST"])

test_folder = dir_functions.make_output_dir(p["TEST_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["CHECKPOINT_TO_TEST"])
test_folder = os.path.join(test_folder, f"checkpoint_{p['CHECKPOINT_TO_TEST']}")
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

checkpoint_dir = os.path.join(output_folder, "checkpoints")

# Checkpoint to test
cp_max = p["CHECKPOINT_TO_TEST"]

#--------------------     LOAD DATA     --------------------#

data = np.load(p["DATAFILE"])
u, xt, g_u = data['u'], data['xt'], data['g_u']

#-------------------- DATA PREPROCESSING --------------------#

data_p = preprocess_data(u, xt, g_u, resample_i=p["RESAMPLE_TIME_INTERVAL"], nr_samples=p["NR_SAMPLES"], nr_timesteps=p["NR_TIMESTEPS"], train_perc=p["TRAIN_PERC"], 
                         train_idx=p["TRAIN_IDX"], val_idx = p["VAL_IDX"], test_idx=p["TEST_IDX"],
                         u_scaler=p["U_SCALER"], xt_scaler=p["XT_SCALER"], g_u_scaler=p["G_U_SCALER"], batch_xt=p["BATCH_TRUNK"],
                         x_2d=p["X_2D"], same_scale_in_out=p["SAME_SCALE_IN_OUT"], residual=p["RESIDUAL_IN_T"],
                         add_u0_to_gu=p["ADD_U0_TO_G_U"])

model = load_model(output_folder, cp_max)
model = compile_model(model, learning_rate=p["LEARNING_RATE"], scheduler=p["SCHEDULER"])
print(f"Load the model from checkpoint {cp_max}")

if p["BATCH_TRUNK"]:
    _, g_u_pred_train_proc = eval_in_batches(model, data_p['u_train_trans'], data_p['g_u_train_trans'], data_p['xt_train_trans'], 
                                                    batch_size=p["BATCH_SIZE"], batch_xt=p["BATCH_TRUNK"])
    _, g_u_pred_test_proc = eval_in_batches(model, data_p['u_test_trans'], data_p['g_u_test_trans'], data_p['xt_test_trans'], 
                                                   batch_size=p["BATCH_SIZE"], batch_xt=p["BATCH_TRUNK"])
else:
    if data_p['xt_train_trans'] is None:
        g_u_pred_train_proc = model(data_p['u_train_trans'])
        g_u_pred_test_proc = model(data_p['u_test_trans'])
    else:
        g_u_pred_train_proc = model([data_p['u_train_trans'], data_p['xt_train_trans']])
        g_u_pred_test_proc = model([data_p['u_test_trans'], data_p['xt_test_trans']])

# Plot errors between g_u and predicted g_u by the model
with open(os.path.join(test_folder, "stats.txt"), 'w') as f:
    g_u, g_u_pred = evaluate_data(f, test_folder, data_p['g_u_scaler'], data_p['x_len'], data_p['t_len'], data_p['g_u_train_trans'], g_u_pred_train_proc, data_p['g_u_test_trans'], g_u_pred_test_proc)

plot_full_history(test_folder, os.path.join(output_folder, "training_log.out"), plot_name="full_training_history", checkpoint=cp_max)

shape = (g_u.shape[0], data_p['t_len'], int(g_u.shape[1]/data_p['t_len']))
g_u = g_u.reshape(shape)
g_u_pred = g_u_pred.reshape(shape)

# Animate wave and predicted wave on top of each other
for i in p["SAMPLES_TO_ANIMATE"]:
    a.animate_wave(g_u[i], g_u_pred=g_u_pred[i], save_name=os.path.join(test_folder, f"animation_sample_{i}"))