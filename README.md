# DON-LSTM
**DON-LSTM**: Multi-Resolution Learning with DeepONets and Long Short-Term Memory Neural Networks.

This proposed architecture leverages the two main advantages of its subnetworks:
- **DeepONets**: The discretization-invariant training,
- **LSTMs**: The improved sequential learning through memory-preserving mechanisms.

This repository contains scripts that allows for the generation of samples based on a specified equation, and the training of a model based on the generated samples.

## Installation

The software has been developed with and tested for Python 3.11.

```shell
pip install -r requirements.txt
```

## Data Generation
The data generation script is run with a specified equation (kdv is written as an example):

```shell
python generate_data.py kdv
```

The output is a .npz file containing the samples.

There are three equations included, with the possibility of adding more by inheriting the Datagen class from data_generation.py:

1. **kdv** The Korteweg-de Vries equation.
2. **bbm** The Benjamin-Bona-Mahony equation.
3. **cahnhill** The Cahn-Hilliard equation.

### The Data

The data used for training is required to be an .npz file. When loaded as `data` it should have the following structure:
- `data["g_u"]`: the trajectories of the solution, i.e., a timeseries of the shape `[batch, t, x]` or `[batch, t * x]`,
- `data["u"]`: the initial values, i.e., a timeseries of the shape `[batch, x]`
- `data["xt"]`: the locations at which the solution is defined (locations of `data["g_u"]`)

### Configuration

The data generation is configured with the params_datagen.yml file.

* **seed**: Sets the seed for the random generation of the samples. 
* **nr_realizations**: The number of realizations/samples to generate. 
* **folder**: The folder to store the data. 
* **filename**: The name of the .npz file to store the data in. 

## Network training

### Training a Vanilla DeepONet and DON-LSTM

The initial script to train a vanilla DeepONet and DON-LSTM is run by doing:

```shell
python train_model.py
```

This script is configured through the file `params_model.yml`. More information on each setting can be found in the **Configuration** section below.

The model is output in a specified folder from the configuration file. The folder will contain a checkpoints folder that is output in a specified frequency from the configuration file. Once training is complete, plots will be output showing the error over time.

### Training of the DON-LSTM using a Pre-Trained DON

After training a model, it is possible to use it to train the DON-LSTM. This is done by doing:

```shell
python train_don_add_lstm.py
```

This script is configured through the file `params_don_add_lstm.yml`. The model is outputed in a similar way to the previous script.

To use this, load a previously trained model using the **LOADED_MODEL** setting. Be sure that **MODEL_FOLDER**, **PROBLEM_NAME**, and **N_HIGH_RES** are the same from when the loaded model was being trained. **LOAD_CP** chooses which checkpoint of the loaded, pre-trained model to start from. If `null`, then the program will automatically choose the best checkpoint to use using the lowest validation error.

### Model Testing

A model can be tested by running:

```shell
python test_model.py
```

This script is configured through the file `params_test.yml`.

Test results are output in a folder specified by the **TEST_FOLDER** setting. A checkpoint can be chosen from the model by the **CHECKPOINT_TO_TEST** setting. The outputs include training history plots, a text file `stats.txt` that outputs the MAE, RMSE, and RSE errors when the model is being tested on the training and testing data. An animation with both the real solution and the predicted solution by the model can be created by denoting which samples to animate in the array for the **SAMPLES_TO_ANIMATE** setting.

### Configuration

* **PROBLEM_NAME**: Denotes the equation/problem name.
* **N_HIGH_RES**: The number of realizations/samples in the data. 
* **MODEL_FOLDER**: The folder to store the data. 
* **MODELNAME**: The name of the model. 

The model being trained will be stored in a folder `{MODEL_FOLDER}/{PROBLEM_NAME}_{N_HIGH_RES}_samples/{MODELNAME}`.

* **BATCH_SIZE**: The size of each batch to train on.
* **TRAIN_IDX**: The range of indices for the training data. 
* **VAL_IDX**: The range of indices for the validation data. 
* **TEST_IDX**: The range of indices for the testing data. 

The indices must not exceed the number of samples in the data. It is recommended that the testing data not overlap with the training or validation data.

* **START_FROM_CHECKPOINT**: If the model already exists, the program can start from an existing checkpoint to continue to train from. The checkpoint must exist in the model's checkpoint folder. If the value is 0, it will create a new model from scratch.
* **N_EPOCHS**: The total number of epochs to train the model for. 
* **CHECKPOINTS_FREQ**: Save a checkpoint after every specified interval of epochs. 
* **LOG_OUTPUT_FREQ**: Log results after every specified interval of epochs. 