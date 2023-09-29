"""
File to train DON-LSTM starting from a pre-trained DeepONet model.
Parameters are read from the params_don_add_lstm.py file.
The pre-trained model is loaded from the folder specified in params_don_add_lstm.py or through passed parameter "--model" (the passed parameter overwrites the parameters file).
"""

import sys
sys.path.insert(0, './modules')
import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.keras import Model
from modules.training import train_model, eval_in_batches
from modules.plotting import plot_history_all, plot_rmse_in_time
from modules.model_definition import make_nn, compile_model, add_layers
from modules.data_manipulation import preprocess_data, postprocess_data, scaling, subset_indices, subset_array, make_xt_from_idx
from modules.load_model import load_model
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions
import params_don_add_lstm as params
from tensorflow.keras.layers import Dense, Reshape


#--------------------     PARAMS FOR TESTING     --------------------#

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', required=False, action="store_true",
                    help="Run only 2 epochs and print the losses; save to a test folder.")
parser.add_argument('-m', '--model', required=False, help="Read a model and continue training from the last epoch.")
parser.add_argument('-p', '--problem', required=False, help="Problem name.")
parser.add_argument('-mn', '--modelname', required=False, help="Model name used in saving.")
parser.add_argument('-ns', '--number_high_res_samples', required=False, help="Number of high-resolution samples used in training.")
parser.add_argument('-mi', '--model_idx', required=False, help="Index of the low res. model to be used.")

args = parser.parse_args()
in_testing, model = args.test, args.model

if args.problem is not None:
    params.PROBLEM_NAME = args.problem
if args.modelname is not None:
    params.MODELNAME = args.modelname
if args.number_high_res_samples is not None:
    params.N_HIGH_RES = int(args.number_high_res_samples)
if args.model_idx is not None:
    params.MODEL_INT = int(args.model_idx)


dir_models = params.DIR_LOAD_MODEL 
cp = params.LOAD_CP

if in_testing:
    N_EPOCHS_1 = 5
    N_EPOCHS_2 = 5
    LOG_OUTPUT_FREQ = 1
    CP_FREQ = 1
else:
    N_EPOCHS_1 = params.N_EPOCHS_1
    N_EPOCHS_2 = params.N_EPOCHS_2
    LOG_OUTPUT_FREQ = params.LOG_OUTPUT_FREQ
    CP_FREQ = params.CHECKPOINTS_FREQ

# Params
evaluation_batches = False
bool_sample_t = (params.SAMPLING_PERC_T < 1.0) and (params.SAMPLING_PERC_T is not None) and (params.SAMPLING_PERC_T!="None")
bool_sample_x = (params.SAMPLING_PERC_X < 1.0) and (params.SAMPLING_PERC_X is not None) and (params.SAMPLING_PERC_X!="None")

bool_batch_trunk = False
if params.TRUNK_RNN:
    bool_batch_trunk = True
if bool_sample_t or bool_sample_x:
    bool_batch_trunk = True

#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(params, is_testing=in_testing)
# Directories
checkpoint_dir = os.path.join(output_folder, "checkpoints")
if not params.START_FROM_LATEST:
    cp_max = 0
    logs_path = os.path.join(output_folder, "log.out")
    params_log_path = os.path.join(output_folder, "params.txt")
    training_log_path_0 = os.path.join(output_folder, "training_log_0.out")
    training_log_path_1 = os.path.join(output_folder, "training_log_1.out")

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
logs.write(f"\nTraining extended model on {params.LOADED_MODEL}\n")

logs.write(log_functions.print_params(params, table=True, header=True))


#--------------------     LOAD DATA     --------------------#

data = np.load(params.DATAFILE)
u, xt, g_u = data['u'], data['xt'], data['g_u']

u, g_u = u[params.I0_SAMPLE:], g_u[params.I0_SAMPLE:]

if params.NR_SAMPLES is not None:
    u, g_u = u[:params.NR_SAMPLES], g_u[:params.NR_SAMPLES]

if params.OPT_LAMBDA_INIT_COND!=0:
    params.ADD_U0_TO_G_U = True

if params.OPT_HARD_CONSTR_INT or (params.OPT_LAMBDA_ALL_TS > 0) or (params.OPT_LAMBDA_INIT_COND > 0) or (params.OPT_LAMBDA_LAST_TS > 0):
    params.SAME_SCALE_IN_OUT = True


#-------------------- DATA PREPROCESSING --------------------#

data_p = preprocess_data(u, xt, g_u, resample_i=params.RESAMPLE_I, nr_timesteps=params.NR_TIMESTEPS, train_perc=params.TRAIN_PERC,
                         train_idx=params.TRAIN_IDX, val_idx = params.VAL_IDX, test_idx=params.TEST_IDX,
                         u_scaler=params.U_SCALER, xt_scaler=params.XT_SCALER, g_u_scaler=params.G_U_SCALER, trunk_rnn=params.TRUNK_RNN, batch_xt=bool_batch_trunk,
                         wt_input=params.WT_INPUT, wt_output=params.WT_OUTPUT, wt_method=params.WT_METHOD, wt_mode=params.WT_MODE,
                         split_outputs=params.SPLIT_OUTPUTS, x_2d=params.X_2D, same_scale_in_out=params.SAME_SCALE_IN_OUT, residual=params.RESIDUAL_IN_T,
                         add_u0_to_gu=params.ADD_U0_TO_G_U, scaling_coef_bool=params.HARD_CONSTR_INTEGRAL, target_integ=params.TARGET_INTEG, bias=params.BIAS,)

xt_train, xt_test = data_p['xt_train'], data_p['xt_test']
u_train_transformed, u_test_transformed = data_p['u_train_trans'], data_p['u_test_trans']
g_u_train_transformed, g_u_test_transformed = data_p['g_u_train_trans'], data_p['g_u_test_trans']
xt_train_transformed, xt_test_transformed = data_p['xt_train_trans'], data_p['xt_test_trans']
g_u_scaler = data_p['g_u_scaler']

logs.write(log_functions.print_scale_info(u_train_transformed, u_test_transformed, g_u_train_transformed, g_u_test_transformed, xt_train, xt_test))

#-------------------- DOWNSAMPLE DATA --------------------#

if bool_sample_t or bool_sample_x:
    g_u_train_transformed = g_u_train_transformed.reshape(g_u_train_transformed.shape[0], data_p["t_len"], data_p["x_len"])
    indices_t = subset_indices(g_u_train_transformed, params.SAMPLING_PERC_T, axis=1)
    g_u_train_transformed = subset_array(g_u_train_transformed, indices_t, axis=1)
    indices_x = subset_indices(g_u_train_transformed, params.SAMPLING_PERC_X, axis=2)
    g_u_train_transformed = subset_array(g_u_train_transformed, indices=indices_x, axis=2)
    s = g_u_train_transformed.shape
    g_u_train_transformed = g_u_train_transformed.reshape(s[0], s[1]*s[2])
    xt_train = make_xt_from_idx(indices_t, indices_x)
    xt_train_transformed, _ = scaling(xt_train, data_p['xt_scaler'])


#-------------------- MODEL DEFINITION --------------------#

def _find_best_cp(model_main_folder, modelname, X, g_u_test, cp_max, cp_freq):
    c = 0
    print("Max cp:" + str(cp_max))
    print("cp_freq:" + str(cp_freq))
    for cp in np.arange(cp_freq, cp_max, cp_freq):
        model = load_model(main_folder=model_main_folder, modelname=modelname, checkpoint=cp)
        g_u_test_pred = model(X)
        error_sample = np.sqrt(np.mean((g_u_test-g_u_test_pred)**2))
        if c==0:
            error_lowest = error_sample
            cp_best = cp
            print("First best cp: " + str(cp_best))

        elif error_sample < error_lowest:
            error_lowest = error_sample
            cp_best = cp
            print("New best cp: " + str(cp_best))
        c += 1
    return cp_best

if (cp is None) or (cp=="None"):
    cp_max0 = dir_functions.read_max_checkpoint(checkpoint_dir=os.path.join(dir_models, params.LOADED_MODEL, "checkpoints"))

    cp = _find_best_cp(model_main_folder=dir_models, 
                            modelname=params.LOADED_MODEL, 
                            X=[data_p['u_test_trans'], data_p['xt_test_trans']], 
                            g_u_test=data_p['g_u_test_trans'], 
                            cp_max=cp_max0, cp_freq=params.CHECKPOINTS_FREQ)

params.CP_LOADED_MODEL = cp

model_pretrained = load_model(main_folder=dir_models, modelname=params.LOADED_MODEL, checkpoint=cp)

input_layer, output_layer = model_pretrained.input, model_pretrained.output

x = Reshape((data_p["t_len"],data_p["x_len"]))(output_layer)
x = add_layers(x=x, hidden_layers=params.RNN_LAYERS, nn_name="lstm", l2_norm=params.L2_NORM)
x = Dense(data_p["x_len"], activation='linear')(x)
x = Reshape((data_p["t_len"]*data_p["x_len"],))(x)

for layer in model_pretrained.layers:
    layer.trainable = False

model = Model(inputs=input_layer, outputs=x)


model = compile_model(model, learning_rate=params.LEARNING_RATE_1)
logs.write(log_functions.print_model_summary(model, "deeponet"))
print(model.summary())


with open(os.path.join(output_folder, "model_structure.json"), "w") as json_file:
    json_file.write(model.to_json())

logs.close()
logs = open(logs_path, 'a')

#-------------------- MODEL TRAINING --------------------#

tb_writer = tf.summary.create_file_writer(os.path.join(output_folder, "tb_logs"))
history, log_history = train_model(model,
                                   n_epochs=N_EPOCHS_1,
                                   batch_size=params.BATCH_SIZE,
                                   u=u_train_transformed,
                                   xt=xt_train_transformed,
                                   g_u=g_u_train_transformed,
                                   val_perc=params.VAL_PERC,
                                   val_idx = params.VAL_IDX,
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=params.CHECKPOINTS_FREQ,
                                   log_output_freq=LOG_OUTPUT_FREQ,
                                   log_temp_path=training_log_path_0,
                                   last_cp=cp_max,
                                   batch_xt=bool_batch_trunk,
                                   sa_weights=params.OPT_SA_WEIGHTS,
                                   use_tf_function=params.TRAIN_TF, 
                                   tb_writer=tb_writer)

plot_history_all(history, output_folder, "training_history_0")

print(f"\nWarning: Changing the learning rate from {params.LEARNING_RATE_1} to {params.LEARNING_RATE_2}.\n")

cp_max = N_EPOCHS_1



cp_best = _find_best_cp(model_main_folder=os.path.dirname(output_folder), 
                        modelname=os.path.basename(output_folder), 
                        X=[data_p['u_test_trans'], data_p['xt_test_trans']], 
                        g_u_test=data_p['g_u_test_trans'], 
                        cp_max=cp_max, cp_freq=CP_FREQ)


if in_testing:
    cp_max = N_EPOCHS_1
else:
    cp_max = cp_best

params.CP_0 = cp_max

for layer in model.layers:
    layer.trainable = True

model = compile_model(model, learning_rate=params.LEARNING_RATE_2, split_outputs=params.SPLIT_OUTPUTS)

history, log_history = train_model(model,
                                   n_epochs=N_EPOCHS_2,
                                   batch_size=params.BATCH_SIZE,
                                   u=u_train_transformed,
                                   xt=xt_train_transformed,
                                   g_u=g_u_train_transformed,
                                   val_perc=params.VAL_PERC,
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=params.CHECKPOINTS_FREQ,
                                   log_output_freq=LOG_OUTPUT_FREQ,
                                   log_temp_path=training_log_path_1,
                                   last_cp=cp_max,
                                   batch_xt=bool_batch_trunk,
                                   sa_weights=params.OPT_SA_WEIGHTS,
                                   use_tf_function=params.TRAIN_TF, 
                                   tb_writer=tb_writer)


plot_history_all(history, output_folder, "training_history_1")
logs.write(log_functions.print_training_history(log_history))

#-------------------- MODEL EVALUATION --------------------#

if bool_batch_trunk:
    _, g_u_pred_train_transformed = eval_in_batches(model, u_train_transformed, g_u_train_transformed, xt_train_transformed, 
                                                    batch_size=params.BATCH_SIZE, batch_xt=bool_batch_trunk)
    _, g_u_pred_test_transformed = eval_in_batches(model, u_test_transformed, g_u_test_transformed, xt_test_transformed, 
                                                   batch_size=params.BATCH_SIZE, batch_xt=bool_batch_trunk)
else:
    g_u_pred_train_transformed = model([u_train_transformed, xt_train_transformed])
    g_u_pred_test_transformed = model([u_test_transformed, xt_test_transformed])


params_output_processing = dict({'scaler': g_u_scaler, 'wavelet': params.WT_OUTPUT,
                                 'wt_method': params.WT_METHOD, 'wt_mode': params.WT_MODE,
                                 'data_len': data_p['x_len']})

outputs = [g_u_train_transformed, g_u_pred_train_transformed, g_u_test_transformed, g_u_pred_test_transformed]
outputs = postprocess_data(outputs, **params_output_processing)


logs.write(log_functions.print_testing(g_u_train=outputs[0], 
                                       g_u_train_pred=outputs[1], 
                                       g_u_test=outputs[2], 
                                       g_u_test_pred=outputs[3],
                                       g_u_scaler=None))


with open(params_log_path, 'w') as params_log:
    params_log.write(log_functions.print_params(params))
params_log.close()

plot_rmse_in_time(g_u=outputs[2], g_u_pred=outputs[3].numpy(), t_len=data_p['t_len'], output_folder=output_folder, plot_name="mse_in_time")

logs.write(log_functions.print_time("Program finished"))
logs.close()
