"""
Parameters for training DeepONets, LSTMs and DON-LSTMs. These parameters are loaded by train_model.py.
"""

MODELNAME = "don_lstm_highres"					# str: name of the model
N_HIGH_RES = 250						# int: number of high resolution samples
PROBLEM_NAME = "kdv"						# str: name of the problem


# The training data is chosen based on the problem name.
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

OUTPUT_FOLDER = "multires_models/{}_{}_samples/{}".format(PROBLEM_NAME, N_HIGH_RES, MODELNAME)


BATCH_SIZE = 50							# int: batch size
TRAIN_IDX = (0,N_HIGH_RES)					# tuple(int): start and end indices of the training data
VAL_IDX = (3000,3250)						# tuple(int): start and end indices of the validation data
TEST_IDX = (4000,5000)						# tuple(int): start and end indices of the test data
RESAMPLE_I = None 						# int: resample the data in temporal dimension, by every 'i' time steps

# Training parameters
START_FROM_LATEST = False					# bool: start from the last checkpoint if loading a pre-trained model
LEARNING_RATE = 0.0001						# float: learning rate used in training
N_EPOCHS = 25000						# int: number of training epochs
TRAIN_PERC = 0.9						# float: the percentage of data used for training
VAL_PERC = 0.1							# float: the percentage of training data used for validation
CHECKPOINTS_FREQ = 1000						# int: the frequency of saving the model's weights during training
LOG_OUTPUT_FREQ = 100						# int: the frequency of logs

BRANCH_HIDDEN_LAYERS = [{'dense': {'units': 150, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}, 
			{'dense': {'units': 250, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}, 
			{'dense': {'units': 450, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}, 
			{'dense': {'units': 380, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}, 
			{'dense': {'units': 320, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}]

TRUNK_HIDDEN_LAYERS = [{'dense': {'units': 200, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}},
		       {'dense': {'units': 220, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}},
		       {'dense': {'units': 240, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}},
		       {'dense': {'units': 250, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}},
		       {'dense': {'units': 260, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}},
		       {'dense': {'units': 280, 'activation': 'swish', 'kernel_initializer': 'glorot_normal'}}]

L2_NORM = 0.009							# float: L2 regularization
SCHEDULER = None						# str: name of the learning rate scheduler

# RNN_LAYERS: Set to None if training a vanilla DeepONet. Specify if training a DON-LSTM.
#RNN_LAYERS = None						# list(dict): RNN layers to be added after the einsum layer

RNN_LAYERS = [{'lstm': {'units': 200, 'return_sequences': True, 'activation': 'tanh', 
                        'kernel_initializer': 'glorot_uniform'}}] 
									

TRAIN_TF = False						# bool: use @tf.function(jit_compile=True) in training (advised as True when not using the LSTM layer, otherwise False)
U_SCALER = "standard" 						# str: the name of the scaler of the branch input / LSTM input
XT_SCALER = "minmax" 						# str: the name of the scaler of the trunk input
G_U_SCALER = "standard" 					# str: the name of the scaler of the network output
NR_TIMESTEPS = None						# int: nr timesteps to which the data should be subset
NR_SAMPLES = None						# int: nr samples to which the data should be subset

SAMPLING_PERC_T = 1	 					# float: percent of each sample from training data, in time dimension. Each sample has different locations. 
SAMPLING_PERC_X = 1 						# float: percent of each sample from training data, in time dimension. Each sample has different locations. 
BT_OUTPUT_SIZE = 300 						# int: number of neurons to be added at the end of the branch and trunk layers (must be equal)
OUTPUT_ACTIVATION = "linear"					# str: activation function of the last layer of the branch and trunk networks
X_2D = False							# bool: True if using 2D-space data
OPT_SA_WEIGHTS = True						# bool: use self-adaptive weights in the loss function
RESIDUAL_IN_T = False						# bool: Transform the output into residuals from the initial condition. (or from the previous time step?)
ADD_U0_TO_G_U = True						# bool: Add the initial condition u to the output g_u through concatenation: g_u = u + g_u.
SAME_SCALE_IN_OUT = False					# bool: Apply the same scaler (u_scaler) to both u and g_u.
