import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import seaborn as sns
import os

SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_errors(errors_list, errors_list2=None, errors_list3=None, t=None, save_path=None, models_names=None, models_names2=None, models_names3=None, title="", 
                color=None, color2=None, color3=None, linestyle=None, linewidth=1, ylim=None, fontsize=12, legend=True,
                bbox_to_anchor=(1,1.03), std_list=None, std_list2=None):

    val_min, val_max = (np.min(errors_list), np.min(errors_list2)), (np.max(errors_list), np.max(errors_list2))
    val_min = min((i for i in val_min if i is not None), default=None)
    val_max = max((i for i in val_max if i is not None), default=None)
    # Adding margins
    val_min = val_min - 0.05*(abs(val_max) - abs(val_min))   
    val_max = val_max + 0.05*(abs(val_max) - abs(val_min))

    if ylim is None:
        ylim = (val_min, val_max)

    if errors_list is not None:
        t_len = len(errors_list[0])
    else:
        t_len = len(errors_list2[0])
    if t is None:
        t = range(t_len)
 
    if linestyle is None:
        linestyle = ["-"]*t_len

    # For standard deviation
    t_indexes = np.linspace(0,t_len,5)[1:]
    t_indexes = [int(t-1) for t in t_indexes]

    plt.figure(figsize=(6,3))
    if errors_list is not None:
        for i, errors in enumerate(errors_list):
            if models_names is not None:
                plt.plot(t, errors, label=models_names[i], linestyle=linestyle[i], c=color, linewidth=linewidth)
                
                if std_list is not None:
                    for t_ind in t_indexes:
                        plt.errorbar(t[t_ind], errors[t_ind], yerr=std_list[i][t_ind], c=color, lw=1, capsize=1)
            else:
                plt.plot(t, errors, label="Model "+str(i+1), linestyle=linestyle[i], c=color, linewidth=linewidth)

    if errors_list2 is not None:
        for i, errors in enumerate(errors_list2):
            if models_names2 is not None:
                plt.plot(t, errors, label=models_names2[i], linestyle=linestyle[i], c=color2, linewidth=linewidth)
                if std_list2 is not None:
                    for t_ind in t_indexes:
                        plt.errorbar(t[t_ind], errors[t_ind], yerr=std_list2[i][t_ind], c=color, lw=1, capsize=1)
            else:
                plt.plot(t, errors, label="Model "+str(i+1), linestyle=linestyle[i], c=color2, linewidth=linewidth)
    
    if errors_list3 is not None:
        for i, errors in enumerate(errors_list3):
            if models_names3 is not None:
                plt.plot(t, errors, label=models_names3[i], linestyle=linestyle[i+1], c=color3, linewidth=linewidth)

            else:
                plt.plot(t, errors, label="Model "+str(i+1), linestyle=linestyle[i+1], c=color3, linewidth=linewidth)

    if legend:
        if bbox_to_anchor is not None:
            plt.legend(bbox_to_anchor=bbox_to_anchor)
        else:
            plt.legend(loc="upper left")
    plt.xlabel("Time", fontdict={'fontsize': fontsize})
    plt.ylabel("Error (MAE)", fontdict={'fontsize': fontsize})
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontdict={'fontsize': fontsize})
    if ylim is not None:
        plt.ylim(ylim)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_2d_error(mean_error_xt, save_path=None, xticklabels="auto", yticklabels="auto"):
    """
    Plot the mean error for each time and space point.
    """
    plt.figure()
    sns.heatmap(mean_error_xt, vmin=0, vmax=0.1, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel("space")
    plt.ylabel("time")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_by_ts(predictions, model_names, t, sample_i, save_name, folder, modelnames_dict=None, ext=".png", label_size=8.5):


    nr_models = len(model_names)
    timestamps = np.linspace(0,len(t),5)[1:]
    timestamps = [int(t-1) for t in timestamps]
    t_len = len(t)
    x_len = int(predictions[0]['g_u_test_pred'].shape[1]/t_len)
    
    fig, ax = plt.subplots(len(timestamps), nr_models, sharex=True, sharey=True, figsize=(9, 4.5))

    for col_index, pred in enumerate(predictions):
        g_u_test, g_u_test_pred = pred['g_u_test'], pred['g_u_test_pred']

        # Plots
        g_u_test_resh = g_u_test.reshape(g_u_test.shape[0], t_len, x_len)
        g_u_test_pred_resh = g_u_test_pred.reshape(g_u_test_pred.shape[0], t_len, x_len)
        
        for row_index, ts in enumerate(timestamps):
            if col_index==0:
                ax[row_index, col_index].set_ylabel(f"u(x, t={np.round(t[ts],2)})", fontdict={'fontsize': label_size})#, rotation=0)
                ax[row_index, col_index].set_yticks([])
            if row_index==0:
                if modelnames_dict is not None:
                    ax[row_index, col_index].set_title(f"{modelnames_dict[model_names[col_index]]}", fontdict={'fontsize': label_size})
                else:
                    ax[row_index, col_index].set_title(f"{model_names[col_index]}", fontdict={'fontsize': label_size})

            if row_index==(len(timestamps)-1):
                ax[row_index, col_index].set_xlabel(f"x", fontdict={'fontsize': label_size})
            ax[row_index,col_index].set_xticks([])
            ax[row_index,col_index].plot(g_u_test_resh[sample_i, ts], c="blue", linewidth=1, label="Ground truth")
            ax[row_index,col_index].plot(g_u_test_pred_resh[sample_i, ts], c="red", linewidth=1, label="Prediction", linestyle="--")
        
        

    plt.savefig(os.path.join(folder, f"{save_name}{ext}"), bbox_inches="tight")
    plt.close()


def plot_history_all(history, output_folder, plot_name):
    plot_history(history, logy=False, savedir=os.path.join(output_folder, plot_name+".png"))
    plot_history(history, logy=True, savedir=os.path.join(output_folder, plot_name+"_log.png"))
    plot_history(history, logy=False, plot_train=False, savedir=os.path.join(output_folder, plot_name+"_val.png"))
    plot_history(history, logy=True, plot_train=False, savedir=os.path.join(output_folder, plot_name+"_val_log.png"))


# TODO: change this to plot differently
def plot_rmse_in_time(g_u, g_u_pred, t_len, output_folder, plot_name,
                      ylabel="RMSE = $\sqrt{(y-\hat{y})^2}$"):
    shape = (g_u.shape[0], t_len, int(g_u.shape[1]/t_len))
    g_u, g_u_pred = g_u.reshape(shape), g_u_pred.reshape(shape)
    mse_by_ts = np.mean(np.mean(np.sqrt((g_u_pred-g_u)**2), axis=2), axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(mse_by_ts)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title("Prediction error in time")
    plt.savefig(os.path.join(output_folder, plot_name+".png"))
    plt.close()

def plot_history(history, logx=False, logy=False, savedir=None, plot_val=True, plot_train=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    if plot_train:
        ax.plot(history['train_loss'], label="Training loss")
    if plot_val:
        ax.plot(history['val_loss'], label="Validation loss")
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.legend()
    if savedir is not None:
        plt.savefig(savedir)
    else:
        plt.show()

    
def plot_weights(weights, t_len, epoch, savefolder="plots_weights"):
    if os.path.exists(savefolder) == False:
        os.makedirs(savefolder)
    plt.figure(figsize=(15,3))
    weights = weights.read_value().numpy()
    weights = weights.reshape((int(weights.shape[0]/t_len), t_len), order="F")
    plt.imshow(weights, cmap='RdBu_r')
    plt.title("Loss weights")
    plt.ylabel("Spatial points (x)")
    plt.xlabel("Timesteps (t)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, f"plot_epoch_{epoch}.png"))
    plt.close()
    
def plot_solution(y, y_pred, num_list, epoch, int_start=1, delta_plot_idx=50, savefolder="plots_training/training_data", savename=None):
    if os.path.exists(savefolder) == False:
        os.makedirs(savefolder)

    plt.figure(figsize=(10, len(num_list)*2))
    for j in range(int_start, int_start+len(num_list)):
        plt.subplot(len(num_list), 1, j)
        for k, num in enumerate(num_list):
            plt.axvline(x=num, c="lightgrey", linestyle="--")
            if (j==int_start) and (k==0):
                plt.plot(list(range(num,num+delta_plot_idx)), y[j, num:(num+delta_plot_idx)], label=f"True", c="g", linestyle="--")
                plt.plot(list(range(num,num+delta_plot_idx)), y_pred[j, num:(num+delta_plot_idx)], label=f"Predicted", c="r", linestyle="-")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            else:
                plt.plot(list(range(num,num+delta_plot_idx)), y[j, num:(num+delta_plot_idx)], c="g", linestyle="--")
                plt.plot(list(range(num,num+delta_plot_idx)), y_pred[j, num:(num+delta_plot_idx)], c="r", linestyle="-")
            plt.title(f"$\int$ pred: {np.sum(y_pred[j, num:(num+delta_plot_idx)]):.3f}, $\int$ truth: {np.sum(y[j, num:(num+delta_plot_idx)]):.3f}")
        
    plt.tight_layout()
    if savename is None:
        plt.savefig(os.path.join(savefolder, f"plot_epoch_{epoch}.png"))
    else:
        plt.savefig(os.path.join(savefolder, f"{savename}.png"))
    plt.close()