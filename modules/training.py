import numpy as np
import os
import warnings
import tensorflow as tf
from timeit import default_timer
from modules.data_manipulation import train_test_split


def train_model(model, n_epochs, batch_size, u, g_u, xt=None, val_perc=None, val_idx=None, checkpoints_folder="checkpoints",
                checkpoints_freq=1000, log_output=True, log_output_freq=100, log_temp_path="training_log.out", last_cp=0, 
                batch_xt=False, 
                sa_weights=True, shuffle=False, use_tf_function=True, use_batch_remainder=False, tb_writer=None):
    """
    Shapes:
        @ u:   [bs, x_len]
        @ xt:  [bs, x_len * t_len, 3] or [x_len * t_len, 2]
        @ g_u: [bs, x_len * t_len]
    """
    train_loss_epoch, val_loss_epoch = np.array([], dtype=np.float16), np.array([], dtype=np.float16)
    checkpoint_path = os.path.join(checkpoints_folder, "cp-{epoch:04d}.ckpt")
    log_temp = open(log_temp_path, 'w')

    dirname = os.path.dirname(checkpoints_folder)

    if (val_perc is not None) and (val_idx is None):
        u, g_u, u_val, g_u_val, xt, xt_val = train_test_split(
                u=u, g_u=g_u, xt=xt, train_perc=1-val_perc, batch_xt=batch_xt)
    else:
        last_idx = val_idx[1]-val_idx[0]
        u, g_u, u_val, g_u_val = u[:-last_idx], g_u[:-last_idx], u[-last_idx:], g_u[-last_idx:]
        xt_val = xt
        if batch_xt:
            xt, xt_val = xt[:-last_idx], xt[-last_idx:]

    # Initialize self-adaptive weights for the loss
    if sa_weights:
        size = g_u.shape[1]
        loss_weights = tf.Variable(tf.cast(np.ones(size), dtype="float32"), name="loss_weights")
        model.loss_weights = loss_weights
        model.loss_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            
    else:
        model.loss_weights = None
        model.loss_weights_integral = None

    u, g_u = tf.cast(u, tf.float32), tf.cast(g_u, tf.float32)
    if xt is not None:
        xt = tf.cast(xt, tf.float32)

    for i in range(last_cp, last_cp+n_epochs):

        t1 = default_timer()
        train_loss_minibatch = 0
        n_batches = int(np.floor((u.shape[0]+1) / batch_size))
        if use_batch_remainder:
            n_batches = n_batches + 1            

        if shuffle:
            indices = tf.random.shuffle(tf.range(u.get_shape()[0]))
            u, g_u = tf.gather(u, indices), tf.gather(g_u, indices)

        for i1 in np.arange(batch_size, u.shape[0]+1, batch_size):
            i0 = i1 - batch_size
            u_i, g_u_i, xt_i = subset_batch(u=u, g_u=g_u, xt=xt, i0=i0, i1=i1, batch_xt=batch_xt)
            if xt is None:
                X = u_i
            else:
                X = [u_i, xt_i]
            
            if use_tf_function:
                train_loss_i, y_pred = tf_grad_predict(model, X, g_u_i, sa_weights=sa_weights)
            else:
                train_loss_i, y_pred = grad_predict(model, X, g_u_i, sa_weights=sa_weights)
            
            train_loss_minibatch += train_loss_i
        train_loss_minibatch /= n_batches

        train_loss_epoch = np.append(train_loss_epoch, train_loss_minibatch).astype("float16")
        if tb_writer is not None:
            with tb_writer.as_default():
                tf.summary.scalar('train_loss', train_loss_minibatch, step=i)


        if val_perc is not None:
            val_loss_minibatch, val_pred = eval_in_batches(
                model, u=u_val, g_u=g_u_val, xt=xt_val, batch_size=batch_size, batch_xt=batch_xt)
            val_loss_epoch = np.append(val_loss_epoch, val_loss_minibatch).astype("float16")

        else:
            val_loss_epoch = np.append(val_loss_epoch, 0)

        if tb_writer is not None:
            with tb_writer.as_default():
                tf.summary.scalar('val_loss', val_loss_minibatch, step=i)

        t2 = default_timer()

        # LOG
        if log_output:
            log_string = ""
        if i % log_output_freq == 0:
            log_epoch = "epoch: {epoch}\t time: {time:.5f}\t train_mse: {train_mse:.10f}\t val_mse: {val_mse:.10f}\t integral_mse: {int_mse}".format(
                epoch=i, time=(t2-t1), train_mse=train_loss_minibatch, val_mse=val_loss_minibatch, int_mse="NaN")
            print(log_epoch)

            with open(log_temp_path, 'a') as log_temp:
                log_temp.write(f"{log_epoch}\n")
            log_temp.close()

            if log_output:
                log_string += log_epoch + "\n"

        if (i+1) % checkpoints_freq == 0:
            model.save_weights(checkpoint_path.format(epoch=(i+1)))

    history = dict({'train_loss': train_loss_epoch, 'val_loss': val_loss_epoch})

    if log_output:
        return history, log_string

    else:
        return history


def grad_predict_no_sa_weights(model, X, g_u_i):

    with tf.GradientTape(persistent=True) as tape:
        train_loss_i, y_pred = compute_loss(model, X=X, y=g_u_i) 

    gradients = tape.gradient(train_loss_i, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 

    return train_loss_i, y_pred

def grad_predict_sa_weights(model, X, g_u_i):

    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(X)
        model.loss_weights.assign(model.loss_weights/tf.reduce_mean(model.loss_weights))
        train_loss_i = weighted_mse_loss(g_u_i, y_pred, model.loss_weights)

    gradients = tape.gradient(train_loss_i, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
    model.loss_optimizer.apply_gradients(zip([-tape.gradient(train_loss_i, model.loss_weights)], [model.loss_weights]))

    return train_loss_i, y_pred
   

def subset_batch(u, g_u, xt=None, i0=0, i1=256, batch_xt=False):
    """
    Subsets u, g_u and x_t to the indices (i0,i1).
    """

    i0, i1 = int(i0), int(i1)

    def _subset(f, i0, i1):
        if isinstance(f, (tuple, list)):
            f_subset = [f_i[i0:i1] for f_i in f]
        else:
            f_subset = f[i0:i1]
        return f_subset
    
    u_i, g_u_i = _subset(u, i0, i1), _subset(g_u, i0, i1)

    if (batch_xt and (xt is not None)):
        xt_i = xt[i0:i1]
    elif xt is not None:
        xt_i = xt
    else:
        xt_i = None
    
    return u_i, g_u_i, xt_i
    

def eval_in_batches(model, u, g_u, xt=None, batch_size=256, batch_xt=False):
    """
    Evaluates the model on provided (u,xt)-> g_u and returns the average MSE. 
    """
    # TODO: doesn't add the remainder of the data if batch_size is too big
    loss_list = list()
    pred_list = list()

    if batch_size > u.shape[0]:
        if xt is None:
            X = u
        else:
            X = [u, xt]
        loss, pred = compute_loss(model, X, g_u)
        pred_list.extend(pred)
        loss_list.append(loss.numpy())
        warnings.warn("batch_size larger than the size of the validation data.", UserWarning)

    for i1 in np.arange(batch_size, u.shape[0]+1, batch_size):
        i0 = i1 - batch_size
        u_i, g_u_i, xt_i = subset_batch(
            u, g_u, xt, i0, i1, batch_xt=batch_xt)
        if xt is None:
            loss, pred = compute_loss(model, u_i, g_u_i)
        else:
            loss, pred = compute_loss(model, [u_i, xt_i], g_u_i)
        pred_list.extend(pred)
        loss_list.append(loss.numpy()) #TODO
    loss = np.mean(loss_list)
    predictions = np.array(pred_list)

    return loss, predictions



def grad_predict(model, X, g_u_i, sa_weights=False):
    """
    Computes the gradient of the loss function with respect to the trainable variables of the model and updates the model weights.
    """
    if sa_weights:
        train_loss_i, y_pred = grad_predict_sa_weights(model, X, g_u_i)
    else:
        train_loss_i, y_pred = grad_predict_no_sa_weights(model, X, g_u_i)

    return train_loss_i, y_pred


# LOSS FUNCTIONS

def weighted_mse_loss(y_true, y_pred, loss_weights):
    loss = tf.reduce_mean(tf.multiply(loss_weights, tf.square(y_true-y_pred)))
    return loss

def compute_loss(model, X, y):
    """
    Compute loss on the model prediction.
    """

    y_pred = model(X)
    loss = model.loss(y_pred, y)

    return loss, y_pred


# TF FUNCTIONS

@tf.function(jit_compile=True)
def tf_grad_predict_sa_weights(model, X, g_u_i):
    return grad_predict_sa_weights(model, X, g_u_i)

@tf.function(jit_compile=True)
def tf_grad_predict_no_sa_weights(model, X, g_u_i):
    return grad_predict_no_sa_weights(model, X, g_u_i)

def tf_grad_predict(model, X, g_u_i, sa_weights=False):
    """
    Computes the gradient of the loss function with respect to the trainable variables of the model and updates the model weights.
    """
    if sa_weights:
        train_loss_i, y_pred = tf_grad_predict_sa_weights(model, X, g_u_i)
    else:
        train_loss_i, y_pred = tf_grad_predict_no_sa_weights(model, X, g_u_i)

    return train_loss_i, y_pred
