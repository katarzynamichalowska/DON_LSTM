"""
File to train DeepONets, LSTMs and DON-LSTMs. 
Parameters are read from the params.py file.
"""
# TODO: Sys imports are hardcoded.
import sys
sys.path.insert(0, './modules')
import numpy as np
import os
from modules.training import train_model, eval_in_batches
from modules.plotting import plot_history_all, plot_rmse_in_time
from modules.model_definition import make_deeponet, make_nn, compile_model, make_modified_rnn
from modules.data_manipulation import preprocess_data, postprocess_data
from modules.load_model import load_model
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions
from modules.testing import compute_metrics


#--------------------     PARAMS FOR TESTING     --------------------#

p, in_testing = dir_functions.load_params('params_model.yml')

if in_testing:
    N_EPOCHS = 5
    LOG_OUTPUT_FREQ = 1
else:
    N_EPOCHS = p["N_EPOCHS"]
    LOG_OUTPUT_FREQ = p["LOG_OUTPUT_FREQ"]

# Params
bool_sample_t = (p["SAMPLING_PERC_T"] is not None) and (p["SAMPLING_PERC_T"] < 1.0)
bool_sample_x = (p["SAMPLING_PERC_X"] is not None) and (p["SAMPLING_PERC_X"] < 1.0)
bool_batch_trunk = True if bool_sample_t or bool_sample_x else False

#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(p["MODEL_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["START_FROM_CHECKPOINT"], is_testing=in_testing)

checkpoint_dir = os.path.join(output_folder, "checkpoints")

# Log names
cp_max = p["START_FROM_CHECKPOINT"]
logs_path = os.path.join(output_folder, "log.out")
params_log_path = os.path.join(output_folder, "params.txt")
training_log_path = os.path.join(output_folder, "training_log.out")

# Write params to param log
log_functions.log_params(params_log_path, p)

logs = open(logs_path, 'w')
logs.write(log_functions.print_time("Program started"))
logs.write(log_functions.print_params(p, table=True, header=True))


#--------------------     LOAD DATA     --------------------#

data = np.load(p["DATAFILE"])
u, xt, g_u = data['u'], data['xt'], data['g_u']

#-------------------- DATA PREPROCESSING --------------------#

data_p = preprocess_data(u, xt, g_u, resample_i=p["RESAMPLE_TIME_INTERVAL"], nr_samples=p["NR_SAMPLES"], nr_timesteps=p["NR_TIMESTEPS"], train_perc=p["TRAIN_PERC"], 
                         train_idx=p["TRAIN_IDX"], val_idx = p["VAL_IDX"], test_idx=p["TEST_IDX"],
                         u_scaler=p["U_SCALER"], xt_scaler=p["XT_SCALER"], g_u_scaler=p["G_U_SCALER"], batch_xt=bool_batch_trunk,
                         x_2d=p["X_2D"], same_scale_in_out=p["SAME_SCALE_IN_OUT"], residual=p["RESIDUAL_IN_T"],
                         add_u0_to_gu=p["ADD_U0_TO_G_U"])

#logs.write(log_functions.print_scale_info(data_p['u_train_trans'], data_p['u_test_trans'], data_p['g_u_train_trans'], data_p['g_u_test_trans'], data_p['xt_train'], data_p['xt_test']))

#-------------------- MODEL DEFINITION --------------------#

# TODO: This has to be thought through actually... Create a "ModelFactory" function?
# Define an example model of a deeponet and a separate model of an RNN in a separate file?

if not p["START_FROM_CHECKPOINT"]:

    if (p["TRUNK_HIDDEN_LAYERS"] is not None) and (p["BRANCH_HIDDEN_LAYERS"] is not None):


        branch_network = make_nn(input_shape=data_p['u_train_trans'].shape[1:], 
                                hidden_layers=p["BRANCH_HIDDEN_LAYERS"], 
                                output_size=p["BT_OUTPUT_SIZE"],
                                output_activation=p["OUTPUT_ACTIVATION"],
                                nn_name="branch",
                                l2_norm=p["L2_NORM"])

        trunk_network = make_nn(input_shape=data_p['xt_train_trans'].shape[1:], 
                                hidden_layers=p["TRUNK_HIDDEN_LAYERS"], 
                                output_size=p["BT_OUTPUT_SIZE"],
                                output_activation=p["OUTPUT_ACTIVATION"],
                                nn_name="trunk",
                                l2_norm=p["L2_NORM"])

        model = make_deeponet(branch_network=branch_network,
                            trunk_network=trunk_network,
                            rnn_layers=p["RNN_LAYERS"],
                            x_len=data_p['x_len'],
                            t_len=data_p['t_len'],
                            trunk_batch=bool_batch_trunk)
        

        logs.write(log_functions.print_model_summary(branch_network, "branch"))
        logs.write(log_functions.print_model_summary(trunk_network, "trunk"))
        logs.write(log_functions.print_model_summary(model, "deeponet"))
    
    elif (p["RNN_LAYERS"] is not None):  
        model = make_modified_rnn(input_shape=data_p['u_train_trans'].shape[1:], rnn_layers=p["RNN_LAYERS"], 
                             x_len=data_p['x_len'], t_len=data_p['t_len'], l2_norm=p["L2_NORM"])

        logs.write(log_functions.print_model_summary(model, "rnn"))

        data_p['xt_train_trans'], data_p['xt_test_trans'] = None, None

    model = compile_model(model, learning_rate=p["LEARNING_RATE"], scheduler=p["SCHEDULER"])


    with open(os.path.join(output_folder, "model_structure.json"), "w") as json_file:
        json_file.write(model.to_json())

    logs.close()
    logs = open(logs_path, 'a')

else:
    model = load_model(output_folder, cp_max)
    model = compile_model(model, learning_rate=p["LEARNING_RATE"], scheduler=p["SCHEDULER"])
    print(f"Load the model from checkpoint {cp_max}")

#-------------------- MODEL TRAINING --------------------#

# TODO create wrapper for train model including plot_name and plot_history_all

history, log_history = train_model(model,
                                   n_epochs=N_EPOCHS,
                                   batch_size=p["BATCH_SIZE"],
                                   u=data_p['u_train_trans'],
                                   xt=data_p['xt_train_trans'],
                                   g_u=data_p['g_u_train_trans'],
                                   val_perc=p["VAL_PERC"],
                                   val_idx = p["VAL_IDX"],
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=p["CHECKPOINTS_FREQ"],
                                   log_output_freq=LOG_OUTPUT_FREQ,
                                   log_temp_path=training_log_path,
                                   last_cp=cp_max,
                                   batch_xt=bool_batch_trunk,
                                   sa_weights=p["OPT_SA_WEIGHTS"],
                                   use_tf_function=p["TRAIN_TF"])


plot_name = "training_history"
plot_history_all(history, output_folder, plot_name)
logs.write(log_functions.print_training_history(log_history))

#-------------------- MODEL EVALUATION --------------------#

if bool_batch_trunk:
    _, g_u_pred_train_proc = eval_in_batches(model, data_p['u_train_trans'], data_p['g_u_train_trans'], data_p['xt_train_trans'], 
                                                    batch_size=p["BATCH_SIZE"], batch_xt=bool_batch_trunk)
    _, g_u_pred_test_proc = eval_in_batches(model, data_p['u_test_trans'], data_p['g_u_test_trans'], data_p['xt_test_trans'], 
                                                   batch_size=p["BATCH_SIZE"], batch_xt=bool_batch_trunk)
else:
    if data_p['xt_train_trans'] is None:
        g_u_pred_train_proc = model(data_p['u_train_trans'])
        g_u_pred_test_proc = model(data_p['u_test_trans'])
    else:
        g_u_pred_train_proc = model([data_p['u_train_trans'], data_p['xt_train_trans']])
        g_u_pred_test_proc = model([data_p['u_test_trans'], data_p['xt_test_trans']])

params_output_processing = dict({'scaler': data_p['g_u_scaler'], 'data_len': data_p['x_len']})

outputs = postprocess_data([data_p['g_u_train_trans'], g_u_pred_train_proc, data_p['g_u_test_trans'], g_u_pred_test_proc], **params_output_processing)


metrics_train = compute_metrics(outputs[0], outputs[1])
metrics_test = compute_metrics(outputs[2], outputs[3])

logs.write(log_functions.print_testing(metrics_train=metrics_train, metrics_test=metrics_test))

plot_rmse_in_time(g_u=outputs[2], g_u_pred=outputs[3].numpy(), t_len=data_p['t_len'], output_folder=output_folder, plot_name="rmse_in_time")

logs.write(log_functions.print_time("Program finished"))
logs.close()
