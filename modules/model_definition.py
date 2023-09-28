import sys
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Lambda, Conv1D, Flatten, MaxPooling1D, AveragePooling1D,
                                     Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Dropout, Concatenate, 
                                     SimpleRNN, LSTM, GRU, LeakyReLU)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from keras.utils.generic_utils import get_custom_objects
#from tensorflow.keras.saving import get_custom_objects


def dict_to_layers(hidden_layers):
    return [LAYER_DICT[list(layer.keys())[0]](**list(layer.values())[0]) for layer in hidden_layers]

def hybridpool_1d(name, i, control_size=False, **parameters):
    """
    Hybrid pooling: concatenation of max and average pooling layers.
    Params:
        @control_size: apply a convolutional layer at the end
                       otherwise the output size is doubled (relative to regular pooling)
    """
    layer_name = f"{name}_layer{i}"
        
    def hybrid(x):
        hybridpool = Concatenate(name=f"{layer_name}c")([
            MaxPooling1D(**parameters, name=f"{layer_name}a")(x),
            AveragePooling1D(**parameters, name=f"{layer_name}b")(x)])
        if control_size:
            hybridpool = Conv1D(int(x.shape[-1]), 1, name=f"{layer_name}d")(hybridpool)
            
        return hybridpool
    
    return hybrid


def hybridpool_2d(name, i, control_size=False, **parameters):
    
    def hybrid(x):
        layer_name = f"{name}_layer{i}"
        
        hybridpool = Concatenate(name=f"{layer_name}c")([
            MaxPooling2D(**parameters, name=f"{layer_name}a")(x),
            AveragePooling2D(**parameters, name=f"{layer_name}b")(x)])
        if control_size:
            hybridpool = Conv2D(int(x.shape[-1]), (1, 1), name=f"{layer_name}d")(hybridpool)
            
        return hybridpool
    
    return hybrid

LAYER_DICT = dict({"dense": Dense, 
                   "conv_1d": Conv1D,
                   "conv_2d": Conv2D,
                   "maxpool_1d": MaxPooling1D,
                   "avgpool_1d": AveragePooling1D,                       
                   "maxpool_2d": MaxPooling2D,
                   "avgpool_2d": AveragePooling2D,
                   "hybridpool_1d": hybridpool_1d,
                   "hybridpool_2d": hybridpool_2d,
                   "reshape": Reshape,
                   "flatten": Flatten,
                   "dropout": Dropout,
                   "rnn": SimpleRNN, 
                   "lstm": LSTM,
                   "gru": GRU,
                   "leaky_relu": LeakyReLU
                  })


def sine_activation(x):
    """
    Sine activation function.
    """
    return K.sin(x)

get_custom_objects().update({'sine': sine_activation})


def make_deeponet(branch_network,
               trunk_network,
               rnn_layers=None,
               x_len=50,
               t_len=600,
               trunk_batch=False,
               l2_norm_in_rnn=None):

    """
    Creates a DeepONet.
    """
    
    # Einsum trunk - branch
    if not trunk_batch:
        assert(branch_network.output_shape[1]==trunk_network.output_shape[1])
        tb_einsum = Lambda(lambda x: tf.einsum('Np,np->Nn',x[0], x[1]), name="einsum")([branch_network.output, trunk_network.output])

    else:
        assert(branch_network.output_shape[1]==trunk_network.output_shape[2])
        tb_einsum = Lambda(lambda x: tf.einsum('Np,Nnp->Nn',x[0], x[1]), name="einsum")([branch_network.output, trunk_network.output])        

    model = Model(inputs=(branch_network.input, trunk_network.input), outputs=tb_einsum, name="deeponet")
    if rnn_layers is not None:
        model = add_rnn(model=model, rnn_layers=rnn_layers, x_len=x_len, t_len=t_len, l2_norm=l2_norm_in_rnn)

    return model


def make_modified_rnn(input_shape, rnn_layers, x_len, t_len, l2_norm=None):
    layers = [{'dense': {'units': x_len*t_len, 'activation': 'linear'}},
            {'reshape': {'target_shape': (t_len, x_len)}}]
    layers += rnn_layers
    layers +=[{'dense': {'units': x_len, 'activation': 'linear'}},
            {'reshape': {'target_shape': (t_len*x_len,)}}]

    model = make_nn(input_shape=input_shape, hidden_layers=layers, nn_name="rnn", l2_norm=l2_norm)  

    return model

def add_rnn(model, rnn_layers, x_len=50, t_len=200, l2_norm=None):
    """
    Returns a model with additional RNN layers after einsum.
    """
    layers = [{'reshape': {'target_shape': (t_len, x_len)}}]
    layers+= rnn_layers
    layers+= [{'dense': {'units': x_len, 'activation': 'linear'}},
              {'reshape': {'target_shape': (t_len*x_len,)}}]
    x = add_layers(x=model.output, hidden_layers=layers, nn_name="deeponet", l2_norm=l2_norm)
    model_with_rnn = Model(inputs=model.input, outputs=x)

    return model_with_rnn    

def make_rnn(input_shape, rnn_layers, x_len, t_len, nn_name="rnn", l2_norm=None):
    """
    Makes an rnn.
    """
    input_layer = Input(shape=input_shape, name=nn_name+"_input")
    x = input_layer

    layers = [{'reshape': {'target_shape': (t_len, x_len)}}]
    layers+= rnn_layers
    layers+= [{'dense': {'units': x_len, 'activation': 'linear'}},
              {'reshape': {'target_shape': (t_len*x_len,)}}]
    x = add_layers(x=x, hidden_layers=layers, nn_name=nn_name, l2_norm=l2_norm)
    model = Model(inputs=input_layer, outputs=x)

    return model    


def make_nn(input_layer=None,
            input_shape=None, 
            hidden_layers=None, 
            output_size=None,
            output_activation=None,
            conv2d_map_width=12,
            nn_name="nn",
            l2_norm=None):
    """
    Creates a neural network.
    params:
        @ hidden_layers: a list of layers: "dense", "conv1" and "conv2"
        @ kernel_size: kernel_size for the convolutional layers
        @ activation_output: activation function to use in the output layer
        
    Template:
        make_nn(
            input_shape = 50,
            hidden_layers = ["dense": {"units": 60, 
                                       "activation": "tanh"}, 
                             "conv_1d": {"filters": 32, 
                                       "kernel_size": 2, 
                                       "strides": 1,
                                       "activation": "relu"},
                             "maxpool_1d": {"pool_size": 2}
                             "dense": {"units": 70, 
                                       "activation": "tanh"}],
            output_size = 150,
            output_activation = "linear",
            nn_name = "branch"
        )
    
    """
    
    layer_1 = list(hidden_layers[0].keys())[0]
    
    # Define the input layer
    if input_layer is None:
        if layer_1=="conv_1d":
            input_shape = input_shape + (1,)
            input_layer = Input(shape=input_shape, name=nn_name+"_input")
            x = input_layer
        
        elif layer_1=="conv_2d":
            input_layer = Input(shape=input_shape, name=nn_name+"_input")
            x = Dense(conv2d_map_width**2, name=f"{nn_name}_input_expand")(input_layer)
            x = Reshape((conv2d_map_width,conv2d_map_width,1), name=f"{nn_name}_input_2dmap")(x)
        else:
            input_layer = Input(shape=input_shape, name=nn_name+"_input")
            x = input_layer
    else:
        x = input_layer

        
    # Add hidden layers
    x = add_layers(x=x, hidden_layers=hidden_layers, nn_name=nn_name, l2_norm=l2_norm)

    if output_size is None:
        output_layer = x
        output_layer._name = f"{nn_name}_output"
    else:
        output_layer = Dense(output_size, activation=output_activation, name=f"{nn_name}_output")(x)
    
    nn = Model(inputs=input_layer, outputs=output_layer)
    
    return nn

def compile_model(model, learning_rate, scheduler=None):
    if scheduler=="cosine_decay_restarts":
        scheduler_params = dict(initial_learning_rate=learning_rate, first_decay_steps=500, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(**scheduler_params)

    # TODO: here add losses to the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)

    return model


def add_layers(x, hidden_layers, nn_name, l2_norm=None):
    
    layers, parameters = zip(*[(list(d.keys())[0], list(d.values())[0]) for d in hidden_layers])

    # TODO: other layers to add l2 norm

    if l2_norm is not None:
        [p.update({'kernel_regularizer':l2(l2_norm)}) for l, p in zip(layers, parameters) if (l=="dense") or (l=="conv_2d")]
    
    for i, (layer_str, p) in enumerate(zip(layers, parameters)):
        
        p = set_default_parameters(layer_str, p)
        
        layer_name = f"{nn_name}_layer{i+1}"
        
        layer = LAYER_DICT[layer_str]
        
        if (layer_str=="hybridpool_1d") or (layer_str=="hybridpool_2d"):
            x = layer(name=nn_name, i=i+1, **p)(x)
        else:
            x = layer(name=layer_name, **p)(x)            

    
    return x


def set_default_parameters(layer, p):
    
    if layer=="dense":
        p.setdefault("units", 128)
        p.setdefault("activation", "linear") #set to "linear" to be able to add a custom activation function afterwards
        
    elif layer=="conv_1d":
        p.setdefault("filters", 32)
        p.setdefault("kernel_size", 2)
        p.setdefault("strides", 1)
        p.setdefault("activation", "relu")
        
    elif (layer=="maxpool_1d") or (layer=="avgpool_1d") or (layer=="hybridpool_2d"):
        p.setdefault("pool_size", 2)
        
    elif layer=="conv_2d":
        p.setdefault("filters", 32)
        p.setdefault("kernel_size", (2,2))
        p.setdefault("strides", (1,1))
        p.setdefault("activation", "relu")
        
    elif (layer=="maxpool_2d") or (layer=="avgpool_2d") or (layer=="hybridpool_2d"):
        p.setdefault("pool_size", (2,2))
        
    return p

