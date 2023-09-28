# Paths and names
MODELNAME = "model"
DATAFILE = "data/data.npz"
OUTPUT_FOLDER = "models/{}".format(MODELNAME)

# Training parameters
START_FROM_LATEST = False			# Type: bool
TRUNK_RNN = False					# Type: bool
LEARNING_RATE = 0.0001				# Type: float
N_EPOCHS = 1000						# Type: int
BATCH_SIZE = 264					# Type: int
TRAIN_PERC = 0.8					# Type: float
VAL_PERC = 0.2						# Type: float
CHECKPOINTS_FREQ = 100				# Type: int
LOG_OUTPUT_FREQ = 10

# Data parameters
U_SCALER = "standard" 				# Type: str
XT_SCALER = "minmax" 				# Type: str
G_U_SCALER = "standard" 			# Type: str
RESAMPLE_I = None 					# Type: int
NR_TIMESTEPS = None					# Type: int or None

# Branch network parameters

BRANCH_HIDDEN_LAYERS = [{'dense': {'units': 200, 'activation': 'swish'}}, 
						{'dense': {'units': 400, 'activation': 'swish'}}, 
						{'reshape': {'target_shape': (20, 20, 1)}}, 
						{'conv_2d': {'filters': 64, 'activation': 'swish', 'strides': (1, 1), 'kernel_size': 7}}, 
						{'hybridpool_2d': {'pool_size': (3, 3)}}, 
						{'conv_2d': {'filters': 128, 'activation': 'swish', 'strides': (1, 1), 'kernel_size': 3}}, 
						{'hybridpool_2d': {'pool_size': (2, 2)}}, 
						{'dense': {'units': 400, 'activation': 'swish'}}, 
						{'dense': {'units': 500, 'activation': 'swish'}}, 
						{'dense': {'units': 520, 'activation': 'swish'}}, 
						{'dense': {'units': 360, 'activation': 'swish'}}, 
						{'dense': {'units': 280, 'activation': 'swish'}}]


BRANCH_CONV2D_MAP_WIDTH = 20 		# Type: int

# Trunk network parameters
TRUNK_HIDDEN_LAYERS = [      
                        {'dense': {'units': 200, 'activation': 'swish'}},
	                    {'dense': {'units': 220, 'activation': 'swish'}},
	                    {'dense': {'units': 240, 'activation': 'swish'}},
	                    {'dense': {'units': 250, 'activation': 'swish'}},
	                    {'dense': {'units': 260, 'activation': 'swish'}},
	                    {'dense': {'units': 280, 'activation': 'swish'}}]
    
TRUNK_CONV2D_MAP_WIDTH = 12			# Type: int
L2_NORM = 0.0009				# Type: float or None
SPLIT_OUTPUTS = False
X_2D = False						# Type: bool

RNN_LAYERS = None#[
				#{'rnn': {'units': 100, 'return_sequences': True}}] 
									# Type: list(dict) or None

# Model parameters
BT_OUTPUT_SIZE = 300 				# Type: int
OUTPUT_ACTIVATION = "linear"		# Type: str

# Notes
NOTES = ""		# Type: str
