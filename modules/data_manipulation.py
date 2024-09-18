import numpy as np
import sys
sys.path.insert(0, "../")


"""--------------------   PROCESSING (GENERAL)   --------------------"""

def append_g_u_by_u(g_u, u):
    """
    Appends u0 to g_u.
    """
    # TODO: Rename and describe. The functions adds u0 (initial condition) to the output of the network g_u. This means that the timestep 0 is also predicted by the model
    g_u_temp = g_u.reshape(g_u.shape[0], int(g_u.shape[1]/u.shape[1]), u.shape[1])
    u_temp = u.reshape(u.shape[0], 1, u.shape[1])
    g_u = np.concatenate([u_temp, g_u_temp], axis=1)
    g_u = g_u.reshape(g_u.shape[0], g_u.shape[1]*g_u.shape[2])
    
    return g_u

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
    u_scaler = u_scaler.casefold()
    xt_scaler = xt_scaler.casefold()
    g_u_scaler = g_u_scaler.casefold()

    x_len = u.shape[1]
    t_len = int(g_u.shape[1]/x_len)

    if nr_samples is not None:
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

    # resample - skip every resample_i number of timesteps (downsampling)
    if resample_i is not None:
        g_u, xt = resample_g_xt(g_u, xt, int(resample_i))
        t_len = int(t_len/resample_i)
        print(f"Output resampled by i={resample_i}.")
        if add_u0_to_gu:
            t_len += 1


    if nr_timesteps is not None:
        if nr_timesteps >= t_len:
            raise ValueError(f"nr_timesteps={nr_timesteps} is larger than number of timesteps in the data ({t_len})")
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
                                                                                )

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

    # TODO: Create separate dictionaries of training and testing data and scalers? 
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


"""--------------------   SUBSETTING/RESAMPLING   --------------------"""

def resample_g_xt(g_u, xt, i, except_i=False):
    """
    Resamples g_u and xt to a lower resolution.
    Assumption: u and g_u are at the same x locations.
    
    Params:
        @ i:            i for every ith timestamp to be sampled
        @ except_i:     sample every timestep that is not i (for testing)
    """
    
    # g_u:
    g_u_reshaped = g_u.reshape(g_u.shape[0],
                                             len(np.unique(xt[:, 1])),
                                             len(np.unique(xt[:, 0])))
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

    if scaler_type is None:
        f_scaled = f
        scaler_type = None
        scaler_features = dict({})

    elif scaler_type == "standard":
        f_mean, f_std = (f.mean(), f.std()) if isinstance(
            scaler, str) else (scaler['mean'], scaler['std'])
        f_scaled = (f - f_mean) / f_std
        scaler_features = dict({"mean": f_mean, "std": f_std})

    elif scaler_type == "minmax":
        f_min, f_max = (f.min(), f.max()) if isinstance(
            scaler, str) else (scaler['min'], scaler['max'])
        f_scaled = (f - f_min) / (f_max - f_min)
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
            x = (z * scaler['std']) + scaler['mean']

        elif scaler['scaler'] == "minmax":
            x = (z * (scaler['max'] - scaler['min'])) + scaler['max']

        elif scaler['scaler'] == "None":
            x = z

        return x

    if isinstance(z, (list, tuple)):
        # TODO: reformulate the condition: doesn't work if the output is a full np array / tensor
        x = [_inverse_scaling(z_i, scaler_i) for z_i, scaler_i in zip(z, scaler)]
    else:
        x = _inverse_scaling(z, scaler)

    return x

