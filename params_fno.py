# Paths and names
MODELNAME = "fno_lstm"
#DATAFILE = "/gpfs/data/gk/kmichalo/data/kdv/data_kdv.npz"
DATAFILE = "/gpfs/data/gk/kmichalo/data/burgers/data_burgers_nu0.001_nx100.npz"
#OUTPUT_FOLDER = "/gpfs/data/gk/kmichalo/models/fno_models_5000epochs/{}".format(MODELNAME)
OUTPUT_FOLDER = "/gpfs/data/gk/kmichalo/multires_models/fno_burgers_100_samples/{}".format(MODELNAME)

LEARNING_RATE = 0.001			                	# Type: float
N_EPOCHS = 1500					                	# Type: int
BATCH_SIZE = 50					                    # Type: int, datasize%batch==0
TRAIN_PERC = 0.9				                	# Type: float
TRAIN_IDX = (0,100)    # None				        # Type: tuple
VAL_PERC = None  #                           # Type: float
VAL_IDX = (1000,1250)					                	# Type: tuple
TEST_IDX = (2000,2500)					                	# Type: tuple
CHECKPOINTS_FREQ = 100			                	# Type: int
LOG_OUTPUT_FREQ = 10
RESAMPLE_I = None 				                	# Type: int
RNN_LAYER = 200                                       # Type: int (nr neurons)
NR_TIMESTAMPS = None                                # Type: int or None
INPUT_3D = False                                    # Type: bool
LSTM = True
GRU = False

# FNO params
MODES = 8                                          # Type: int
WIDTH = 64                                         # Type: int

# Scheduler parameters
STEP_SIZE = 100                                    # Type: int
GAMMA = 0.5                                        # Type: float