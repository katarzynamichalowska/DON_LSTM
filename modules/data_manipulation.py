import numpy as np
import os
import warnings
from modules.dir_functions import read_parameters, read_params_module_as_dict
import sys
sys.path.insert(0, "../")


"""--------------------   PROCESSING (GENERAL)   --------------------"""


def update_params_with_default(params):
    """
    Updates params with default values if missing.
    """
    # Default values for incomplete params.txt
    default_values = read_params_module_as_dict('params_default')

    # Update dictionary with default values if missing
    [params.update({key:value}) for key, value in default_values.items() if key not in params]
    
    return params

def read_model_params(modelname, folder="../../models", filename="params.txt"):
    """
    Returns:
        @ data_processing_params: a dictionary of parameters in lowercase to be used as kwargs 
                                  in preprocess_data() and postprocess_data()
    """

    params_path = os.path.join(os.path.join(folder, modelname), filename)
    params_read = read_parameters(params_path)
    params_read = dict([(key.upper(),value) for key, value in params_read.items()]) #Keys to uppercase
    params_updated = update_params_with_default(params_read)
    data_processing_params = dict([(k.lower(), v) for k,v in zip(params_updated.keys(), params_updated.values())])

    return params_updated, data_processing_params

def preprocess_data_from_params(u, xt, g_u, modelname, folder="../../models", **fixed_params):
    """
    Wrapper for preprocess_data() which reads parameters from logs (params.txt of a given model)

    Params:
        @ fixed_params: preprocessing parameters with a fixed value (not from file)
    """
 
    params_updated, preprocessing_params = read_model_params(modelname, folder=folder)

    # Update with fixed_params if needed
    [preprocessing_params.update({key:value}) for key, value in fixed_params.items()]
    [params_updated.update({key.upper():value}) for key, value in fixed_params.items()]

    return params_updated, preprocess_data(u, xt, g_u, **preprocessing_params)

def append_g_u_by_u(g_u, u):
    """
    Appends u0 to g_u.
    """
    g_u_temp = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/u.shape[1]), u.shape[1])
    u_temp = u.reshape(u.shape[0], 1, u.shape[1])
    g_u = np.concatenate([u_temp, g_u_temp], axis=1)
    g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])
    
    return g_u

def preprocess_data(u, xt, g_u, resample_i=None, nr_timesteps=None, nr_samples=None, train_perc=0.8, 
                    u_scaler=None, xt_scaler=None, g_u_scaler=None, 
                    batch_xt=False, x_2d=False, same_scale_in_out=False, 
                    residual=False, add_u0_to_gu=True, 
                    train_idx=None, val_idx=None, test_idx=None, **kwargs):
    """
    Data preprocessing: 
        - Resamples u and g_u to a lower resolution,
        - Transforms trunk input for batching,
        - Splits the data into training and testing,
        - Scales the data.
    """

    x_len = u.shape[1]
    t_len = int(g_u.shape[1]/x_len)

    if (nr_samples is not None) and (nr_samples != "None"):
        u, g_u = u[:nr_samples], g_u[:nr_samples]
        if batch_xt:
            xt = xt[:nr_samples]

    if add_u0_to_gu:
        g_u = append_g_u_by_u(g_u, u)
        t_len += 1
        xt = make_xt(x_len=x_len, t_len=t_len)
        xt[:,1] -= 1

    if residual:
        g_u_temp = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/u.shape[1]), u.shape[1])
        g_u_temp[:,1:,:] = g_u_temp[:,1:,:] - g_u_temp[:,:-1,:]
        g_u_temp[:,0,:] = g_u_temp[:,0,:] - u
        g_u = g_u_temp.reshape(g_u.shape[0], g_u.shape[1])

    if x_2d:
        # TODO: not compatible with resampling
        xt = make_xt(x_len, t_len, x_2d=x_2d)

    if (resample_i is not None) and (resample_i != "None"):
        g_u, xt = resample_g_xt(g_u, xt, int(resample_i))
        t_len = int(t_len/resample_i)
        print(f"Output resampled by i={resample_i}.")
        if add_u0_to_gu:
            t_len += 1

    # TODO: make sure that resampled data has the same scaling as regular data (trunk has the same min and max)

    if (nr_timesteps is not None) and (nr_timesteps != "None"):
        if nr_timesteps >= t_len:
            raise ValueError(f"nr_timesteps={nr_timesteps} is larger than number of timestesteps in the data ({t_len})")
        g_u = g_u[:, :nr_timesteps*x_len]
        t_len = int(g_u.shape[1]/x_len)
        xt = make_xt(x_len, t_len, x_2d=x_2d)
        print(f"Output trimmed to {nr_timesteps} timesteps.")
        
    if batch_xt:
        xt = xt.repeat(g_u.shape[0]).reshape(g_u.shape[0], xt.shape[0], xt.shape[1])

    if (train_idx is not None) and (test_idx is not None):
        u_train, g_u_train = u[train_idx[0]:train_idx[1]], g_u[train_idx[0]:train_idx[1]]
        u_val, g_u_val = u[val_idx[0]:val_idx[1]], g_u[val_idx[0]:val_idx[1]]
        u_train, g_u_train = np.concatenate([u_train, u_val], axis=0), np.concatenate([g_u_train, g_u_val], axis=0)
        u_test, g_u_test = u[test_idx[0]:test_idx[1]], g_u[test_idx[0]:test_idx[1]]

        if batch_xt:
            xt_train, xt_val = xt[train_idx[0]:train_idx[1]], xt[val_idx[0]:val_idx[1]]
            xt_train, xt_val = np.concatenate([xt_train, xt_val], axis=0), np.concatenate([xt_train, xt_val], axis=0)
            xt_test = xt[test_idx[0]:test_idx[1]]
        else:
            xt_train, xt_test = xt, xt

    else:        
        u_train, g_u_train, u_test, g_u_test, xt_train, xt_test = train_test_split(u=u,
                                                                                g_u=g_u,
                                                                                xt=xt,
                                                                                train_perc=train_perc,
                                                                                batch_xt=trunk_rnn)

    u_train_trans, xt_train_trans, g_u_train_trans = u_train, xt_train, g_u_train
    u_test_trans, xt_test_trans, g_u_test_trans = u_test, xt_test, g_u_test

    # Scale data. 
    # If (scaler is None) returns the original data
    if train_perc > 0.0:
        u_train_trans, xt_train_trans, g_u_train_trans, u_scaler, xt_scaler, g_u_scaler = scale_data(u=u_train_trans,
                                                                                                    xt=xt_train_trans,
                                                                                                    g_u=g_u_train_trans,
                                                                                                    u_scaler=u_scaler,
                                                                                                    xt_scaler=xt_scaler,
                                                                                                    g_u_scaler=g_u_scaler)
    u_test_trans, xt_test_trans, g_u_test_trans, _, _, _ = scale_data(u=u_test_trans,
                                                                      xt=xt_test_trans,
                                                                      g_u=g_u_test_trans,
                                                                      u_scaler=u_scaler,
                                                                      xt_scaler=xt_scaler,
                                                                      g_u_scaler=g_u_scaler)
    
    if same_scale_in_out:
            g_u_scaler = u_scaler
            g_u_train_trans, _ = scaling(g_u_train, scaler=u_scaler)
            g_u_test_trans, _ = scaling(g_u_test, scaler=u_scaler)


    return dict({
        'u_train': u_train,
        'u_test': u_test,
        'g_u_train': g_u_train,
        'g_u_test': g_u_test,
        'u_train_trans': u_train_trans,
        'u_test_trans': u_test_trans,
        'g_u_train_trans': g_u_train_trans,
        'g_u_test_trans': g_u_test_trans,
        'xt_train': xt_train,
        'xt_test': xt_test,
        'xt_train_trans': xt_train_trans,
        'xt_test_trans': xt_test_trans,
        'u_scaler': u_scaler,
        'g_u_scaler': g_u_scaler,
        'xt_scaler': xt_scaler,
        'x_len': x_len,
        't_len': t_len
        })

def postprocess_data(f, scaler=None, data_len=50, **kwargs):
    """
    Data postprocessing:
    - Unscales the data

    Params:
        @ f: function or a list of functions that should be preprocessed in the same way
    """

    f_post = list()

    if not isinstance(f, list):
        f = [f]

    for f_i in f:

        if scaler is not None:
            f_i = inverse_scaling(z=f_i, scaler=scaler)
        
        f_i = reshape_3d_2d(f_i)
        f_post.append(f_i)
    
    if len(f_post)==1:
        f_post = f_post[0]
    
    return f_post



def postprocess_g_u_from_params(f, modelname, folder, scaler=None, data_len=50):
    """
    Wrapper for preprocess_data() which reads parameters from logs (params.txt of a given model)
    """

    params, data_processing_params = read_model_params(modelname=modelname, folder=folder)


    return params, postprocess_data(f, data_len=data_len, scaler=scaler, **data_processing_params)


def train_test_split(u, g_u, xt=None, train_perc=0.8, batch_xt=False):
    """
    Splits u, xt and g_u into training set.

    Params:
        @ batch_xt: trunk in batches

    if batch_xt:
        @ u.shape = [bs, x_len]
        @ xt.shape = [bs, x_len*t_len, 3]
        @ g_u.shape = [bs, x_len*t_len] 
    else:
        @ u.shape = [bs, x_len]
        @ xt.shape = [x_len*t_len, 2]
        @ g_u.shape = [bs, x_len*t_len] 
    """
    
    def _split(f, train_size):
        """
        Splits f into train and test sets.
        """
        if isinstance(f, (list, tuple)):
            train, test = list(), list()
            for i, f_i in enumerate(f):
                train.append(f_i[:train_size])
                test.append(f_i[train_size:])
                assert(train[i].shape[-1]==test[i].shape[-1])
        else:            
            train, test = f[:train_size], f[train_size:]
            assert(train.shape[-1]==test.shape[-1])

        return train, test

    if train_perc > 0.0:
        train_size = int(np.floor(int(u.shape[0])*train_perc))

        u_train, u_test = _split(u, train_size)
        g_u_train, g_u_test = _split(g_u, train_size)

        if batch_xt:
            xt_train, xt_test = _split(xt, train_size)
        else:
            xt_train, xt_test = xt, xt

        return u_train, g_u_train, u_test, g_u_test, xt_train, xt_test
    
    else:
        return None, None, u, g_u, xt, xt


"""--------------------   SUBSETTING/RESAMPLING   --------------------"""


def subset_ts(u, xt, g_u, ts):
    """
    Subsets the data to the solutions (u.shape[1] spatial points) for a given timestamp.
    """
    x_len = u.shape[1]  
    i0 = ts * x_len
    i1 = i0 + x_len
    xt_i = xt[:, i0:i1, :]
    g_u_i = g_u[:, i0:i1]
    
    return u, xt_i, g_u_i



def resample_g_xt(g_u, xt, i, except_i=False):
    """
    Resamples g_u and xt to a lower resolution.
    Assumption: u and g_u are at the same x locations.
    
    Params:
        @ i:            i for every ith timestamp to be sampled
        @ except_i:     sample every timestep that is not i (for testing)
    """
    
    # g_u:
    g_u_reshaped = reshape_g_u(g_u, xt)
    if not except_i:
        g_u_i = g_u_reshaped[:, ::i]
    else:
        r = np.array(range(g_u_reshaped.shape[1]))
        g_u_i = np.delete(g_u_reshaped, r[::i], axis=1)
    g_u_sampled = g_u_i.reshape(g_u_i.shape[0], g_u_i.shape[1]*g_u_i.shape[2])
    
    # xt:
    xt_reshaped = np.array(
        np.split(xt, np.unique(xt[:, 1], return_index=True)[1][1:]))
    
    if not except_i:
        xt_i = xt_reshaped[::i, :]
    else:
        r = np.array(range(xt_reshaped.shape[0]))
        xt_i = np.delete(xt_reshaped, r[::i], axis=0)

    xt_sampled = xt_i.reshape((xt_i.shape[0]*xt_i.shape[1], xt_i.shape[2]))

    return g_u_sampled, xt_sampled
    
def subset_indices(arr, sub_perc, axis=1):
    """
    Returns an array of indices of shape [bs, sample_size], where sample_size is the length of the array
    across the specified axis * sub_perc.
    Example:
        - arr [bs, x_len, y_len], axis=1: returns [bs, x_len*sub_perc]
        - arr [bs, x_len, y_len], axis=2: returns [bs, y_len*sub_perc]
    """
    s = arr.shape
    indices = np.tile(np.arange(s[axis]), s[0]).reshape(s[0],s[axis])
    if sub_perc < 1.0:
        indices = np.apply_along_axis(np.random.permutation, axis=1, arr=indices)
        indices = indices[:,:int(np.floor(s[axis]*sub_perc))]
        indices = np.apply_along_axis(np.sort, axis=1, arr=indices)

    return indices

def subset_array(arr, indices, axis=1):
    if axis==1:
        arr = np.array([arr_i[idx_i] for arr_i, idx_i in zip(arr,indices)])
    elif axis==2:
        arr = np.array([arr_i[:,idx_i] for arr_i, idx_i in zip(arr,indices)])
    
    return arr

def downsample_x_t(g_u, t_len, x_len, sampling_perc_t, sampling_perc_x):
    g_u = g_u.reshape(g_u.shape[0], t_len, x_len)
    indices_t = subset_indices(g_u, sampling_perc_t, axis=1)
    g_u = subset_array(g_u, indices_t, axis=1)
    indices_x = subset_indices(g_u, sampling_perc_x, axis=2)
    g_u = subset_array(g_u, indices=indices_x, axis=2)
    g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])

    return g_u, indices_t, indices_x


"""--------------------   TRUNK FUNCTIONS   --------------------"""

def make_xt(x_len, t_len, x_2d=False):
    """
    Makes a 2D trunk input of the form:
    x: (x1, x2...xn, x1, x2...xn...xn)
    t: (t1, t1...t1, t2, t2...t2...tn)
    """
    if x_2d:
        x_len = int(np.sqrt(x_len))

    x = np.array(range(1, x_len+1))
    t = np.array(range(1, t_len+1))

    if x_2d:
        x_col1 = np.tile(np.repeat(x, x.shape[0]), t.shape[0])
        x_col2 = np.tile(np.tile(x, x.shape[0]), t.shape[0])
        t_col = np.repeat(t, x.shape[0]**2)
        xt = np.stack([t_col, x_col1, x_col2]).T
    else:
        x_col = np.tile(x, t.shape[0])
        t_col = np.repeat(t, x.shape[0])
        xt = np.stack([x_col, t_col]).T
    
    return xt


def make_xt_from_idx(indices_t, indices_x):
    """
    Returns a batched trunk based in provided indices.
    """
    
    assert(indices_t.shape[0]==indices_x.shape[0])
    
    xt_full = list()
    
    for i in range(indices_t.shape[0]):
        x, t = indices_x[i], indices_t[i]
        x_col = np.tile(x, t.shape[0])
        t_col = np.repeat(t, x.shape[0])
        xt = np.stack([x_col, t_col]).T
        xt_full.append(xt)
    xt_full = np.array(xt_full)
    
    return xt_full


def update_trunk(xt, g_u_pred):
    """
    Replaces the last column in xt with the predicted value.
    
    Params:
        @xt: trunk [bs, x_len*1, 3]
        @g_u_pred: last predicted value of g_u
        
    """
    xt = xt[:, :, :2]
    g_u_pred = g_u_pred.reshape(g_u_pred.shape[0], g_u_pred.shape[1], 1)
    xt = np.concatenate([xt, g_u_pred], axis=2)
    
    return xt


"""--------------------      RESHAPING      --------------------"""

def reshape_3d_2d(g_u):
    """
    Reshape g_u from [bs, t_len, x_len] to [bs, t_len*x_len]
    If the array is not 3D, do nothing.
    """
    def _reshape(g_u):
        if len(g_u.shape)==3:
            g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])
        return g_u
    
    if isinstance(g_u, (list, tuple)):
        g_u = [_reshape(g_u_i) for g_u_i in g_u]
    else:
        g_u = _reshape(g_u)

    return g_u


def reshape_g_u(g_u_output, xt):
    """
    Transforms output functions into an array of shape [bs, t_len, x_len]
    
    params:
        @ g_u_output: DeepONet prediction of shape [bs, t_len * x_len]
        @ xt: trunk input of shape [_,2]
    """

    g_u_output_reshaped = g_u_output.reshape(g_u_output.shape[0],
                                             len(np.unique(xt[:, 1])),
                                             len(np.unique(xt[:, 0])))

    return g_u_output_reshaped


"""--------------------       SCALING       --------------------"""


def scale_data(u, xt, g_u, u_scaler, xt_scaler, g_u_scaler):
    """
    Scaling the data of the format:
        - u: Branch input
        - t: Trunk input
        - g_u: DeepONet output 
    If '*_scaler' is a string: fit and transform; if it is a dictionary, only transform.
    """

    def _multiple_functions(f, scaler):    
        assert(len(f)==len(scaler))      

        f_list, scaler_list = list(), list()
        for f_i, scaler_i in zip(f, scaler):
            f_col, scaler_col = scaling(f_i, scaler_i)
            f_list.append(f_col)
            scaler_list.append(scaler_col)

        return f_list, scaler_list
    
    def _by_column(f, scaler):
        f_list, scaler_list = list(), list()
        for i, scaler_i in enumerate(scaler):
            f_col, scaler_col = scaling(f[:, i], scaler_i)
            f_list.append(f_col)
            scaler_list.append(scaler_col)
        f_scaled = np.stack(f_list).T

        return f_scaled, scaler_list

    
    scaled_functions, scalers = list(), list()

    for i, (f, scaler) in enumerate(zip([u, xt, g_u], [u_scaler, xt_scaler, g_u_scaler])):

        if isinstance(f, (tuple, list)) and (not isinstance(scaler, (tuple, list))):
            scaler = [scaler for i in range(len(f))]

        try:
            if (isinstance(scaler, (dict, str))) or (scaler is None):
                f_scaled, scaler = scaling(f, scaler)

            elif isinstance(scaler, (list, tuple)):
                f_scaled, scaler = _multiple_functions(f, scaler)

            else:
                print("Scaler must be a dictionary or a string, or list of dictionaries and strings.")

        except Exception as err:
            print(f"Error at {i}: {err=}, {type(err)=}")
            raise

        scaled_functions.append(f_scaled)
        scalers.append(scaler)

    u, xt, g_u = scaled_functions
    u_scaler, xt_scaler, g_u_scaler = scalers

    return u, xt, g_u, u_scaler, xt_scaler, g_u_scaler


def scaling(f, scaler):
    """
    Scales f either with standard or minmax scaling.
    @param scaler: str or dict
    """

    scaler_type = scaler if (isinstance(scaler, str) or (scaler is None)) else scaler['scaler']

    if (scaler_type is None) or (scaler_type == "None") or (scaler_type == "none"):
        f_scaled = f
        scaler_type = "None"
        scaler_features = dict({})

    elif scaler_type == "standard":
        f_mean, f_std = (f.mean(), f.std()) if isinstance(
            scaler, str) else (scaler['mean'], scaler['std'])
        f_scaled = standard_scaler(f, f_mean, f_std)
        scaler_features = dict({"mean": f_mean, "std": f_std})

    elif scaler_type == "minmax":
        f_min, f_max = (f.min(), f.max()) if isinstance(
            scaler, str) else (scaler['min'], scaler['max'])
        f_scaled = minmax_scaler(f, f_min, f_max)
        scaler_features = dict({"min": f_min, "max": f_max})

    elif scaler_type == "norm":
        f_norm = np.sqrt(np.mean(f**2, axis=1))
        f_scaled = np.divide(f.T, f_norm).T
        scaler_features = dict({})

    else:
        print("ERROR: Scaler must be either None, \"standard\", \"minmax\" or \"norm\".")

    # Scaler info as a dictionary
    scaler_dict = dict({"scaler": scaler_type})
    scaler_dict.update(scaler_features)

    return f_scaled, scaler_dict


def inverse_scaling(z, scaler):

    def _inverse_scaling(z, scaler):
        if scaler['scaler'] == "standard":
            x = standard_scaler_inverse(z, scaler['mean'], scaler['std'])

        elif scaler['scaler'] == "minmax":
            x = minmax_scaler_inverse(z, scaler['min'], scaler['max'])

        elif scaler['scaler'] == "None":
            x = z

        return x

    if isinstance(z, (list, tuple)):
        # TODO: reformulate the condition: doesn't work if the output is a full np array / tensor
        x = [_inverse_scaling(z_i, scaler_i) for z_i, scaler_i in zip(z, scaler)]
    else:
        x = _inverse_scaling(z, scaler)

    return x

def standard_scaler(x, mean, std):
    """
    Not using StandardScaler because it treats each timestamp as a separate feature.
    """
    z = (x - mean) / std
    return z


def standard_scaler_inverse(z, mean, std):
    x = (z * std) + mean

    return x


def minmax_scaler(x, minimum, maximum):
    z = (x - minimum) / (maximum - minimum)

    return z


def minmax_scaler_inverse(z, minimum, maximum):
    x = (z * (maximum - minimum)) + minimum

    return x

