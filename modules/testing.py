import numpy as np
import pandas as pd
from pprint import pprint


def predict_in_batches(model, u, xt, batch_size, trunk_rnn=False):

    pred_list = list()
    for i1 in np.arange(batch_size, u.shape[0]+1, batch_size):
        i0 = i1 - batch_size

        u_i = u[i0:i1]
        if trunk_rnn:
            xt_i = xt[i0:i1]
        else:
            xt_i = xt

        pred = model([u_i, xt_i])
        pred_list.extend(pred)
    predictions = np.array(pred_list)

    return predictions

#--------------------          Evaluation metrics          --------------------#

def rse(y, y_pred):
    return np.mean((y-y_pred)**2) / np.mean((y-y.mean())**2)

def rmse(y, y_pred):
    return np.sqrt(np.mean((y-y_pred)**2))

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

def compute_metrics(g_u, g_u_pred, round=None, std=False):
    metrics = dict()
    metrics['mae'] = mae(g_u, g_u_pred)
    metrics['rmse'] = rmse(g_u, g_u_pred)
    metrics['rse'] = rse(g_u, g_u_pred)

    if round is not None:
        for key in metrics:
            metrics[key] = np.round(metrics[key], round)

    return metrics


def compute_and_print_metrics(true_train=None, pred_train=None, true_test=None, pred_test=None, round=None):

    compute_train = (true_train is not None) and (pred_train is not None) and (pred_train.shape[0] != 0)
    compute_test = (true_test is not None) and (pred_test is not None)

    if compute_train:
        metrics_train = compute_metrics(true_train, pred_train, round=round)
        metrics_train = pd.DataFrame(metrics_train, index=["train: "])

    if compute_test:
        metrics_test = compute_metrics(true_test, pred_test, round=round)
        metrics_test = pd.DataFrame(metrics_test, index=["test: "])

    if compute_train and compute_test:
        metrics = pd.concat([metrics_train, metrics_test])

    elif compute_train:
        metrics = metrics_train

    elif compute_test:
        metrics = metrics_test

    else:
        raise ValueError(
            "Must provide true and pred for at least training or testing data.")

    pprint(metrics)

    return metrics

def error_in_time(g_u, g_u_pred, x_len=50, metric="mae"):
    if metric=="mae":
        total_error = np.abs(g_u - g_u_pred)
    elif metric=="mse":
        total_error = (g_u - g_u_pred)**2
    total_error = total_error.reshape(total_error.shape[0], int(total_error.shape[1]/x_len), x_len)
    mean_error_xt = np.mean(total_error, axis=0)
    mean_error_t = np.mean(mean_error_xt, axis=1)
    mean_error_t_each_sample = np.mean(total_error, axis=2)
    std_t = np.std(mean_error_t_each_sample, axis=0)
    return mean_error_xt, mean_error_t, std_t


def rse_in_time(g_u, g_u_pred):    
    """
    Computes relative squared error in time.
    g_u and g_u pred have to have shape [bs, t_len, x_len]
    """         
    rse_ts = np.mean(np.mean(((g_u-g_u_pred))**2, axis=2), axis=0) / np.mean(np.mean(((g_u-g_u.mean())**2), axis=2), axis=0)
    return rse_ts