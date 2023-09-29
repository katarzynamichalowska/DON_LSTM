# DON_LSTM
**DON-LSTM**: Multi-Resolution Learning with DeepONets and Long-Short Term Memory Neural Networks.

This proposed architecture leverages the two main advantages of its subnetworks:
- **DeepONets**: The discretization-invariant training,
- **LSTMs**: The improved sequential learning through memory-preserving mechanisms.

## Network training
The files for training the networks:
- `train_model.py`: Training vanilla DeepONets and DON-LSTM
- `train_don_add_lstm.py`: Training of the DON-LSTM using a pre-trained DON

The training parameters of the files are available in:
- `params_model.py`: Parameters for training of the vanilla DeepONets and DON-LSTM
- `params_don_add_lstm.py`: Parameters for training of the DON-LSTM using a pre-trained DON

## Data
The data used for training is required to be an .npz file. When loaded as `data` it should have the following structure:
- `data["g_u"]`: the trajectories of the solution, i.e., a timeseries of the shape [batch, t, x] or [batch, t*x],
- `data["u"]`: the initial values, i.e., a timeseries of the shape [batch, x]
- `data["xt"]`: the locations at which the solution is defined (locations of data["g_u"])
