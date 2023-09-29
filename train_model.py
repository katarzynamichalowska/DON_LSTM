"""
File to train DeepONets, LSTMs and DON-LSTMs. 
Parameters are read from the params.py file.
"""

import sys
sys.path.insert(0, './modules')
import numpy as np
import os
import argparse
from modules.training import train_model, eval_in_batches
from modules.plotting import plot_history_all, plot_rmse_in_time
from modules.model_definition import make_deeponet, make_nn, compile_model, make_modified_rnn
from modules.data_manipulation import preprocess_data, postprocess_data, scaling, make_xt_from_idx, downsample_x_t
from modules.load_model import load_model
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions
import params_model as params
from modules.testing import compute_metrics


#--------------------     PARAMS FOR TESTING     --------------------#

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', required=False, action="store_true")
parser.add_argument('-m', '--model', required=False,
                    help="Read a model and continue training from the last epoch.")
parser.add_argument('-p', '--problem', required=False, help="Problem name.")
parser.add_argument('-mn', '--modelname', required=False, help="Model name used in saving.")
parser.add_argument('-ns', '--number_high_res_samples', required=False, help="Number of high-resolution samples used in training.")

args = parser.parse_args()
in_testing, model = args.test, args.model

if args.problem is not None:
    params.PROBLEM_NAME = args.problem
if args.modelname is not None:
    params.MODELNAME = args.modelname
if args.number_high_res_samples is not None:
    params.N_HIGH_RES = int(args.number_high_res_samples)

if in_testing:
    N_EPOCHS = 5
    LOG_OUTPUT_FREQ = 1
else:
    N_EPOCHS = params.N_EPOCHS
    LOG_OUTPUT_FREQ = params.LOG_OUTPUT_FREQ

# Params
evaluation_batches = False
bool_sample_t = (params.SAMPLING_PERC_T < 1.0) and (params.SAMPLING_PERC_T is not None) and (params.SAMPLING_PERC_T!="None")
bool_sample_x = (params.SAMPLING_PERC_X < 1.0) and (params.SAMPLING_PERC_X is not None) and (params.SAMPLING_PERC_X!="None")
bool_batch_trunk = True if bool_sample_t or bool_sample_x else False

#--------------------       PATHS       --------------------#

output_folder = dir_functions.make_output_dir(params, is_testing=in_testing)
checkpoint_dir = os.path.join(output_folder, "checkpoints")
if not params.START_FROM_LATEST:
    cp_max = 0
    logs_path = os.path.join(output_folder, "log.out")
    params_log_path = os.path.join(output_folder, "params.txt")
    training_log_path = os.path.join(output_folder, "training_log.out")

else:
    cp_max = dir_functions.read_max_checkpoint(checkpoint_dir=checkpoint_dir)
    logs_path = os.path.join(output_folder, f"log_from{cp_max}.out")
    params_log_path = os.path.join(output_folder, f"params_from{cp_max}.txt")
    training_log_path = os.path.join(output_folder, f"training_log_from{cp_max}.out")


with open(params_log_path, 'w') as params_log:
    params_log.write(log_functions.print_params(params))
params_log.close()

logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))
logs.write(log_functions.print_params(params, table=True, header=True))


#--------------------     LOAD DATA     --------------------#

data = np.load(params.DATAFILE)
u, xt, g_u = data['u'], data['xt'], data['g_u']

#-------------------- DATA PREPROCESSING --------------------#

data_p = preprocess_data(u, xt, g_u, resample_i=params.RESAMPLE_I, nr_samples=params.NR_SAMPLES, nr_timesteps=params.NR_TIMESTEPS, train_perc=params.TRAIN_PERC, 
                         train_idx=params.TRAIN_IDX, val_idx = params.VAL_IDX, test_idx=params.TEST_IDX,
                         u_scaler=params.U_SCALER, xt_scaler=params.XT_SCALER, g_u_scaler=params.G_U_SCALER, batch_xt=bool_batch_trunk,
                         x_2d=params.X_2D, same_scale_in_out=params.SAME_SCALE_IN_OUT, residual=params.RESIDUAL_IN_T,
                         add_u0_to_gu=params.ADD_U0_TO_G_U)

xt_train, xt_test = data_p['xt_train'], data_p['xt_test']
u_train_proc, u_test_proc = data_p['u_train_trans'], data_p['u_test_trans']
g_u_train_processed, g_u_test_proc = data_p['g_u_train_trans'], data_p['g_u_test_trans']
xt_train_processed, xt_test_proc = data_p['xt_train_trans'], data_p['xt_test_trans']
g_u_scaler = data_p['g_u_scaler']

logs.write(log_functions.print_scale_info(u_train_proc, u_test_proc, g_u_train_processed, g_u_test_proc, xt_train, xt_test))

#-------------------- DOWNSAMPLE DATA --------------------#

if bool_sample_t or bool_sample_x:
    g_u_train_processed, indices_t, indices_x =  downsample_x_t(g_u=g_u_train_processed, 
                                                                  t_len=data_p["t_len"], x_len=data_p["x_len"], 
                                                                  sampling_perc_t=params.SAMPLING_PERC_T, sampling_perc_x=params.SAMPLING_PERC_X)

    xt_train = make_xt_from_idx(indices_t, indices_x)
    xt_train_processed, _ = scaling(xt_train, data_p['xt_scaler'])


#-------------------- MODEL DEFINITION --------------------#

if not params.START_FROM_LATEST:

    if (params.TRUNK_HIDDEN_LAYERS is not None) and (params.BRANCH_HIDDEN_LAYERS is not None):


        branch_network = make_nn(input_shape=u_train_proc.shape[1:], 
                                hidden_layers=params.BRANCH_HIDDEN_LAYERS, 
                                output_size=params.BT_OUTPUT_SIZE,
                                output_activation=params.OUTPUT_ACTIVATION,
                                nn_name="branch",
                                l2_norm=params.L2_NORM)      

        trunk_network = make_nn(input_shape=xt_train_processed.shape[1:], 
                                hidden_layers=params.TRUNK_HIDDEN_LAYERS, 
                                output_size=params.BT_OUTPUT_SIZE,
                                output_activation=params.OUTPUT_ACTIVATION,
                                nn_name="trunk",
                                l2_norm=params.L2_NORM)

        model = make_deeponet(branch_network=branch_network,
                            trunk_network=trunk_network,
                            rnn_layers=params.RNN_LAYERS,
                            x_len=data_p['x_len'],
                            t_len=data_p['t_len'],
                            trunk_batch=bool_batch_trunk)
        

        logs.write(log_functions.print_model_summary(branch_network, "branch"))
        logs.write(log_functions.print_model_summary(trunk_network, "trunk"))
        logs.write(log_functions.print_model_summary(model, "deeponet"))
    
    elif (params.RNN_LAYERS is not None):  
        model = make_modified_rnn(input_shape=u_train_proc.shape[1:], rnn_layers=params.RNN_LAYERS, 
                             x_len=data_p['x_len'], t_len=data_p['t_len'], l2_norm=params.L2_NORM)

        logs.write(log_functions.print_model_summary(model, "rnn"))

        xt_train_processed, xt_test_proc = None, None

    model = compile_model(model, learning_rate=params.LEARNING_RATE, scheduler=params.SCHEDULER)


    with open(os.path.join(output_folder, "model_structure.json"), "w") as json_file:
        json_file.write(model.to_json())

    logs.close()
    logs = open(logs_path, 'a')

else:
    model = load_model("../models", modelname=model, checkpoint=cp_max)
    print("Load the latest model")

#-------------------- MODEL TRAINING --------------------#


tb_writer = None
history, log_history = train_model(model,
                                   n_epochs=N_EPOCHS,
                                   batch_size=params.BATCH_SIZE,
                                   u=u_train_proc,
                                   xt=xt_train_processed,
                                   g_u=g_u_train_processed,
                                   val_perc=params.VAL_PERC,
                                   val_idx = params.VAL_IDX,
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=params.CHECKPOINTS_FREQ,
                                   log_output_freq=LOG_OUTPUT_FREQ,
                                   log_temp_path=training_log_path,
                                   last_cp=cp_max,
                                   batch_xt=bool_batch_trunk,
                                   sa_weights=params.OPT_SA_WEIGHTS,
                                   use_tf_function=params.TRAIN_TF, 
                                   tb_writer=tb_writer)

if not params.START_FROM_LATEST:
    plot_name = "training_history"
else:
    plot_name = f"training_history_from{cp_max}"

plot_history_all(history, output_folder, plot_name)
logs.write(log_functions.print_training_history(log_history))

#-------------------- MODEL EVALUATION --------------------#

if bool_batch_trunk:
    _, g_u_pred_train_proc = eval_in_batches(model, u_train_proc, g_u_train_processed, xt_train_processed, 
                                                    batch_size=params.BATCH_SIZE, batch_xt=bool_batch_trunk)
    _, g_u_pred_test_proc = eval_in_batches(model, u_test_proc, g_u_test_proc, xt_test_proc, 
                                                   batch_size=params.BATCH_SIZE, batch_xt=bool_batch_trunk)
else:
    if xt_train_processed is None:
        g_u_pred_train_proc = model(u_train_proc)
        g_u_pred_test_proc = model(u_test_proc)
    else:
        g_u_pred_train_proc = model([u_train_proc, xt_train_processed])
        g_u_pred_test_proc = model([u_test_proc, xt_test_proc])

params_output_processing = dict({'scaler': g_u_scaler, 'data_len': data_p['x_len']})

outputs = postprocess_data([g_u_train_processed, g_u_pred_train_proc, g_u_test_proc, g_u_pred_test_proc], **params_output_processing)


metrics_train = compute_metrics(outputs[0], outputs[1])
metrics_test = compute_metrics(outputs[2], outputs[3])

logs.write(log_functions.print_testing(metrics_train=metrics_train, metrics_test=metrics_test))

plot_rmse_in_time(g_u=outputs[2], g_u_pred=outputs[3].numpy(), t_len=data_p['t_len'], output_folder=output_folder, plot_name="rmse_in_time")

logs.write(log_functions.print_time("Program finished"))
logs.close()
