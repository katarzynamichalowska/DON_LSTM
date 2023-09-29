"""
Parameters for training DON-LSTM using a pre-trained DeepONet models. These parameters are loaded by train_don_add_lstm.py.
"""

import os
PROBLEM_NAME = "burgers"
N_HIGH_RES = 700
MODEL_INT = 0

MODELNAME = "don_added_lstm"
OUTPUT_FOLDER = "/gpfs/data/gk/kmichalo/multires_models/{}_{}_samples/{}".format(PROBLEM_NAME, N_HIGH_RES, MODELNAME)
DIR_LOAD_MODEL = "/gpfs/data/gk/kmichalo/multires_models/{}_{}_samples".format(PROBLEM_NAME, N_HIGH_RES)

# Assumes that the preloaded model has "don_lowres" in the name and reads the ith model in the folder (i is MODEL_INT)
LOADED_MODEL = [modelname for modelname in os.listdir(DIR_LOAD_MODEL) if "don_lowres" in modelname][MODEL_INT]


if PROBLEM_NAME=="burgers":
	DATAFILE = "data/burgers/data_burgers_nu0.001.npz"
elif PROBLEM_NAME=="advection":
	DATAFILE = "data/advection/advection.npz"
elif PROBLEM_NAME=="kdv":
	DATAFILE = "data/kdv/data_kdv.npz"
elif PROBLEM_NAME=="diff_sorp":
	DATAFILE = "data/diffusion_sorption/diff_sorp.npz"
elif PROBLEM_NAME=="diff_reac":
	DATAFILE = "data/reaction_diffusion/reac_diff.npz"


# Training parameters
START_FROM_LATEST = False					# bool: 
TRUNK_RNN = False							# bool:
LEARNING_RATE_1 = 0.0001					# float: 
LEARNING_RATE_2 = 0.00001					# float: 
N_EPOCHS_1 = 5000							# int: 
N_EPOCHS_2 = 5000							# int: 
BATCH_SIZE = 10								# int:
TRAIN_PERC = 0.9							# float: 
VAL_PERC = 0.1								# float:
CHECKPOINTS_FREQ = 1000						# int: 
LOG_OUTPUT_FREQ = 100						# int:
LOAD_CP = None                              # int: checkpoint to load, or None if should use the lowest validation error


# Data parameters
U_SCALER = "standard" 						# str:
XT_SCALER = "minmax" 						# str:
G_U_SCALER = "standard" 					# str:
RESAMPLE_I = None 							# int:
NR_TIMESTEPS = None							# int:
NR_SAMPLES = None							# int:
I0_SAMPLE = 0

TRAIN_IDX = (0, N_HIGH_RES)                 # tuple(int): start and end indices of the training data

VAL_IDX = (3000, 3250)



TEST_IDX = (4000,5000)						# tuple(int): start and end indices of the test data

L2_NORM = 0.009								# float: L2 regularization
SCHEDULER = None							# str: learning rate scheduler
#SCHEDULER = "cosine_decay_restarts"	

X_2D = False								# bool: True if using 2D-space data

# Loss function parameters
OPT_LAMBDA_INIT_COND = 0.0					# float: 
OPT_LAMBDA_LAST_TS = 0.0					# float: 
OPT_LAMBDA_ALL_TS = 0.0						# float: 
OPT_HARD_CONSTR_INT = False					# float: 
OPT_LAMBDA_CONSEC_TS_ERROR = 0.0			# float: 
OPT_INT_EVAL_POINTS = None					# list(int): evaluation points for the integral loss. Each integer corresponds to a time step. 
OPT_SA_WEIGHTS = True						# bool: use self-adaptive weights in the loss function


RESIDUAL_IN_T = False						# bool: Transform the output into residuals from the initial condition. (or from the previous time step?)
ADD_U0_TO_G_U = True						# bool: Add the initial condition u to the output g_u through concatenation: g_u = u + g_u.
SAME_SCALE_IN_OUT = False					# bool: Apply the same scaler (u_scaler) to both u and g_u.

# Hard constraints parameters 1
TRAIN_ON_INTEGRALS = False					# bool: ?

# Hard constraints parameters 2
HARD_CONSTR_INTEGRAL = False				# bool: ?
TARGET_INTEG = 10
BIAS = 0.2
UNSCALE_HARD_CONSTR = False					# bool: ?


#RNN_LAYERS = None							# list(dict): RNN layers to be added after the einsum layer
RNN_LAYERS = [{'lstm': {'units': 200, 'return_sequences': True, 'activation': 'tanh', 
                        'kernel_initializer': 'glorot_uniform'}}] 
									

TRAIN_TF = False								# bool: use @tf.function(jit_compile=True) in training

SAMPLING_PERC_T = 1 #0.8 					# float: % of each sample from training data, in time dimension. Each sample has different locations. 
SAMPLING_PERC_X = 1 #0.8					# float: % of each sample from training data, in time dimension. Each sample has different locations. 


BT_OUTPUT_SIZE = 300 						# int: number of neurons to be added at the end of the branch and trunk layers (must be equal)
OUTPUT_ACTIVATION = "linear"				# str: activation function of the last layer of the branch and trunk networks


NOTES = ""									# str: Additional notes to be added to th log file


# Wavelet transform parameters
WT_INPUT = False							# bool: transform the input into wavelets.
WT_OUTPUT = False							# bool: transform the output into wavelets.
WT_METHOD = None							# str: method to use for the wavelet transforms.
WT_MODE = None								# str: mode to use for the wavelet transforms.
SPLIT_OUTPUTS = False						# bool: split the outputs of the wavelet into approximation and detail coefficients.
