import numpy as np
import torch
import sys
sys.path.insert(0, "./modules")
sys.path.insert(0, "./modules/fno")
from modules.fno.utilities3 import UnitGaussianNormalizer


def preprocess_data(u, g_u, S, T, T_in, train_perc=None, val_perc=None, train_idx=None, val_idx=None, test_idx=None, nr_timestamps=None, input_3d=False):

    def _preprocess_u(u, input_3d=input_3d):
        # [bs, x_len, t_len]
        if input_3d:
            u = u.reshape((u.shape[0], int(np.sqrt(u.shape[1])), int(np.sqrt(u.shape[1])), 1))
        else:
            u = u.reshape((u.shape[0], u.shape[1], 1))
        u = torch.tensor(u, dtype=torch.float)
        return u

    def _preprocess_g_u(g_u, x_len, nr_timestamps=nr_timestamps, input_3d=input_3d):
        if input_3d:
            g_u = g_u.reshape((g_u.shape[0], x_len, x_len, int(g_u.shape[1]/(x_len**2))), order="F") # timestamp last
        else:
            #g_u = g_u.reshape((g_u.shape[0], x_len, int(g_u.shape[1]/x_len)), order="F")   # timestamp 3rd
            g_u = g_u.reshape((g_u.shape[0], int(g_u.shape[1]/x_len), x_len))  # timestamp 2nd
        if (nr_timestamps is not None) and (nr_timestamps!="None"):
            g_u = g_u[:, :nr_timestamps, :]

        g_u = torch.tensor(g_u, dtype=torch.float)
        return g_u

    if train_idx is not None:
        print("")
        u_train, g_u_train = u[train_idx[0]:train_idx[1]], g_u[train_idx[0]:train_idx[1]]

    elif train_perc is not None:
        u_train, g_u_train, u_test, g_u_test = train_test(u, g_u, train_perc)

    if input_3d:
        S = int(np.sqrt(S))

    if val_idx is not None:
        u_val, g_u_val = u[val_idx[0]:val_idx[1]], g_u[val_idx[0]:val_idx[1]]

    elif val_perc is not None:
        u_train, g_u_train, u_val, g_u_val = train_test(u_train, g_u_train, 1-val_perc)

    if test_idx is not None:
        u_test, g_u_test = u[test_idx[0]:test_idx[1]], g_u[test_idx[0]:test_idx[1]]
    elif (val_idx is not None) and (train_idx is not None):

        if val_idx[1] > train_idx[1]:
            u_test, g_u_test = u[val_idx[1]:], g_u[val_idx[1]:]
        else:
            u_test, g_u_test = u[train_idx[1]:], g_u[train_idx[1]:]

    u_train, u_val, u_test = _preprocess_u(u_train), _preprocess_u(u_val), _preprocess_u(u_test)
    g_u_train, g_u_val, g_u_test = _preprocess_g_u(g_u_train, S), _preprocess_g_u(g_u_val, S), _preprocess_g_u(g_u_test, S)


    # Normalizing the data
    u_normalizer = UnitGaussianNormalizer(u_train)
    g_u_normalizer = UnitGaussianNormalizer(g_u_train)

    u_train_n, u_val_n, u_test_n = u_normalizer.encode(u_train), u_normalizer.encode(u_val), u_normalizer.encode(u_test)
    g_u_train_n, g_u_val_n, g_u_test_n = g_u_normalizer.encode(g_u_train), g_u_normalizer.encode(g_u_val), g_u_normalizer.encode(g_u_test)

    # Reshape and repeat
    def _reshape_u(u, S, T_in, T, input_3d=input_3d):
        if input_3d:
            u = u.reshape(u.shape[0],S,S,1,T_in).repeat([1,1,1,T,1])

        else:
            #u = u.reshape(u.shape[0],S,1,T_in).repeat([1,1,T,1])    # timestamp 3rd
            u = u.reshape(u.shape[0],1,S,T_in).repeat([1,T,1,1])   # timestamp 2nd
        return u

    if (nr_timestamps is not None) and (nr_timestamps!="None"):
        T = nr_timestamps
   
    u_train = _reshape_u(u=u_train, S=S, T_in=T_in, T=T)
    u_val = _reshape_u(u=u_val, S=S, T_in=T_in, T=T)
    u_test = _reshape_u(u=u_test, S=S, T_in=T_in, T=T)
    u_train_n = _reshape_u(u=u_train_n, S=S, T_in=T_in, T=T)
    u_val_n = _reshape_u(u=u_val_n, S=S, T_in=T_in, T=T)
    u_test_n = _reshape_u(u=u_test_n, S=S, T_in=T_in, T=T)


    return dict({'u_train': u_train, 'u_train_n': u_train_n,
                 'u_val': u_val, 'u_val_n': u_val_n,
                 'u_test': u_test, 'u_test_n': u_test_n,
                 'g_u_train': g_u_train, 'g_u_train_n': g_u_train_n,
                 'g_u_val': g_u_val, 'g_u_val_n': g_u_val_n,
                 'g_u_test': g_u_test, 'g_u_test_n': g_u_test_n,
                 'u_normalizer': u_normalizer,'g_u_normalizer': g_u_normalizer})

def train_test(u, g_u, train_perc):
    train_size = int(np.floor(u.shape[0]*train_perc))
    u_train, g_u_train = u[:train_size], g_u[:train_size]
    u_test, g_u_test = u[train_size:], g_u[train_size:]
    return u_train, g_u_train, u_test, g_u_test