"""
File to train DON-LSTM starting from a pre-trained DeepONet model.
Parameters are read from the params_don_add_lstm.py file.
The pre-trained model is loaded from the folder specified in params_don_add_lstm.py or through passed parameter "--model" (the passed parameter overwrites the parameters file).
"""

import sys
sys.path.insert(0, './')
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model
from modules.training import train_model, eval_in_batches
from modules.plotting import plot_history_all, plot_full_history
from modules.model_definition import compile_model, add_layers
from modules.data_manipulation import preprocess_data
from modules.load_model import load_model
import modules.log_functions as log_functions
import modules.dir_functions as dir_functions
from tensorflow.keras.layers import Dense, Reshape
from modules.evaluate import evaluate_data

#--------------------     PARAMS FOR TESTING     --------------------#

p = dir_functions.load_params(os.path.join('deeponets', 'params_don_add_lstm.yml'))

DIR_LOAD_MODEL = dir_functions.make_output_dir(p["MODEL_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["LOADED_MODEL"], True)

cp = p["LOAD_CP"]

#--------------------       PATHS       --------------------#

# Output folder
output_folder = dir_functions.make_output_dir(p["MODEL_FOLDER"], p["PROBLEM_NAME"], p["N_HIGH_RES"], p["MODELNAME"], p["START_FROM_CHECKPOINT"])

checkpoint_dir = os.path.join(output_folder, "checkpoints")

# Log names
cp_max = p["START_FROM_CHECKPOINT"]
logs_path = os.path.join(output_folder, "log.out")
params_log_path = os.path.join(output_folder, "params.txt")
training_log_path_0 = os.path.join(output_folder, "training_log_0.out")
training_log_path_1 = os.path.join(output_folder, "training_log_1.out")

# Write params to param log
log_functions.log_params(params_log_path, p)

logs = open(logs_path, 'a')
logs.write(log_functions.print_time("Program started"))
logs.write(f"\nTraining extended model on {p['LOADED_MODEL']}\n")
logs.write(log_functions.print_params(p, table=True, header=True))


#--------------------     LOAD DATA     --------------------#

data = np.load(p["DATAFILE"])
u, xt, g_u = data['u'], data['xt'], data['g_u']

u, g_u = u[p["I0_SAMPLE"]:], g_u[p["I0_SAMPLE"]:]

if p["NR_SAMPLES"] is not None:
    u, g_u = u[:p["NR_SAMPLES"]], g_u[:p["NR_SAMPLES"]]

#-------------------- DATA PREPROCESSING --------------------#

data_p = preprocess_data(u, xt, g_u, resample_i=p["RESAMPLE_I"], nr_timesteps=p["NR_TIMESTEPS"], train_perc=p["TRAIN_PERC"],
                         train_idx=p["TRAIN_IDX"], val_idx = p["VAL_IDX"], test_idx=p["TEST_IDX"],
                         u_scaler=p["U_SCALER"], xt_scaler=p["XT_SCALER"], g_u_scaler=p["G_U_SCALER"], batch_xt=p["BATCH_TRUNK"],
                         x_2d=p["X_2D"], same_scale_in_out=p["SAME_SCALE_IN_OUT"], residual=p["RESIDUAL_IN_T"],
                         add_u0_to_gu=p["ADD_U0_TO_G_U"])

logs.write(log_functions.print_scale_info(data_p['u_train_trans'], data_p['u_test_trans'], data_p['g_u_train_trans'], data_p['g_u_test_trans'], data_p['xt_train'], data_p['xt_test']))

#-------------------- MODEL DEFINITION --------------------#

# find best checkpoint of loaded model
def _find_best_cp(model_main_folder, X, g_u_test, cp_max, cp_freq):
    c = 0
    print("Max cp:" + str(cp_max))
    print("cp_freq:" + str(cp_freq))
    cp_best = 0
    for cp in np.arange(cp_freq, cp_max, cp_freq):
        model = load_model(main_folder=model_main_folder, checkpoint=cp)
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

if not p["START_FROM_CHECKPOINT"]:
    # Set up new model
    if cp is None:
        # Get best checkpoint from loaded-in model if checkpoint is not otherwise specified
        cp_max0 = dir_functions.read_max_checkpoint(checkpoint_dir=os.path.join(DIR_LOAD_MODEL, "checkpoints"))

        cp = _find_best_cp(model_main_folder=DIR_LOAD_MODEL, 
                                X=[data_p['u_test_trans'], data_p['xt_test_trans']], 
                                g_u_test=data_p['g_u_test_trans'], 
                                cp_max=cp_max0, cp_freq=p["CHECKPOINTS_FREQ"])

    print(f"Load model {p['LOADED_MODEL']} at checkpoint {cp}")
    model_pretrained = load_model(main_folder=DIR_LOAD_MODEL, checkpoint=cp)
    input_layer, output_layer = model_pretrained.input, model_pretrained.output


    # TODO: Find a more elegant way to add the LSTM layers to the pretrained model.
    x = Reshape((data_p["t_len"],data_p["x_len"]))(output_layer)
    x = add_layers(x=x, hidden_layers=p["RNN_LAYERS"], nn_name="lstm", l2_norm=p["L2_NORM"])
    x = Dense(data_p["x_len"], activation='linear')(x)
    x = Reshape((data_p["t_len"]*data_p["x_len"],))(x)

    # Freezing the pretrained model
    for layer in model_pretrained.layers:
        layer.trainable = False

    model = Model(inputs=input_layer, outputs=x)

    model = compile_model(model, learning_rate=p["LEARNING_RATE_1"])
    logs.write(log_functions.print_model_summary(model, "deeponet"))
    print(model.summary())

    with open(os.path.join(output_folder, "model_structure.json"), "w") as json_file:
        json_file.write(model.to_json())

    logs.close()
    logs = open(logs_path, 'a')

else:
    # Load existing model from checkpoint cp_max
    model = load_model(output_folder, cp_max)
    model = compile_model(model, learning_rate=p["LEARNING_RATE_1"])
    print(f"Load the model from checkpoint {cp_max}")



#-------------------- MODEL TRAINING --------------------#

history, log_history = train_model(model,
                                   n_epochs=p["N_EPOCHS_1"],
                                   batch_size=p["BATCH_SIZE"],
                                   u=data_p['u_train_trans'],
                                   xt=data_p['xt_train_trans'],
                                   g_u=data_p['g_u_train_trans'],
                                   val_perc=p["VAL_PERC"],
                                   val_idx = p["VAL_IDX"],
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=p["CHECKPOINTS_FREQ"],
                                   log_output_freq=p["LOG_OUTPUT_FREQ"],
                                   log_temp_path=training_log_path_0,
                                   last_cp=cp_max,
                                   batch_xt=p["BATCH_TRUNK"],
                                   sa_weights=p["OPT_SA_WEIGHTS"],
                                   use_tf_function=p["TRAIN_TF"])

plot_name = "training_history"
plot_history_all(history, output_folder, plot_name)
logs.write(log_functions.print_training_history(log_history))

print(f"\nWarning: Changing the learning rate from {p['LEARNING_RATE_1']} to {p['LEARNING_RATE_2']}.\n")

cp_max = _find_best_cp(model_main_folder=output_folder,
                        X=[data_p['u_test_trans'], data_p['xt_test_trans']], 
                        g_u_test=data_p['g_u_test_trans'], 
                        cp_max=cp_max, cp_freq=p["CHECKPOINTS_FREQ"])

for layer in model.layers:
    layer.trainable = True

model = compile_model(model, learning_rate=p["LEARNING_RATE_2"])

history, log_history = train_model(model,
                                   n_epochs=p["N_EPOCHS_2"],
                                   batch_size=p["BATCH_SIZE"],
                                   u=data_p['u_train_trans'],
                                   xt=data_p['xt_train_trans'],
                                   g_u=data_p['g_u_train_trans'],
                                   val_perc=p["VAL_PERC"],
                                   checkpoints_folder=checkpoint_dir,
                                   checkpoints_freq=p["CHECKPOINTS_FREQ"],
                                   log_output_freq=p["LOG_OUTPUT_FREQ"],
                                   log_temp_path=training_log_path_1,
                                   last_cp=cp_max,
                                   batch_xt=p["BATCH_TRUNK"],
                                   sa_weights=p["OPT_SA_WEIGHTS"],
                                   use_tf_function=p["TRAIN_TF"])

plot_name = "training_history_1"
plot_history_all(history, output_folder, plot_name)
logs.write(log_functions.print_training_history(log_history))

#-------------------- MODEL EVALUATION --------------------#

if p["BATCH_TRUNK"]:
    _, g_u_pred_train_transformed = eval_in_batches(model, data_p['u_train_trans'], data_p['g_u_train_trans'], data_p['xt_train_trans'], 
                                                    batch_size=p["BATCH_SIZE"], batch_xt=p["BATCH_TRUNK"])
    _, g_u_pred_test_transformed = eval_in_batches(model, data_p['u_test_trans'], data_p['g_u_test_trans'], data_p['xt_test_trans'], 
                                                   batch_size=p["BATCH_SIZE"], batch_xt=p["BATCH_TRUNK"])
else:
    g_u_pred_train_transformed = model([data_p['u_train_trans'], data_p['xt_train_trans']])
    g_u_pred_test_transformed = model([data_p['u_test_trans'], data_p['xt_test_trans']])

evaluate_data(logs, output_folder, data_p['g_u_scaler'], data_p['x_len'], data_p['t_len'], data_p['g_u_train_trans'], g_u_pred_train_transformed, data_p['g_u_test_trans'], g_u_pred_test_transformed)
plot_full_history(output_folder, training_log_path_0, plot_name="full_training_history_0")
plot_full_history(output_folder, training_log_path_1, plot_name="full_training_history_1")

logs.write(log_functions.print_time("Program finished"))
logs.close()
