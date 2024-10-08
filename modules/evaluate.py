from modules.data_manipulation import postprocess_data
from modules.testing import compute_metrics
from modules.log_functions import print_testing
from modules.plotting import plot_rse_in_time

def evaluate_data(log, output_folder, g_u_scaler, x_len, t_len, g_u_train, g_u_pred_train, g_u_test, g_u_pred_test):
    outputs = postprocess_data([g_u_train, g_u_pred_train, g_u_test, g_u_pred_test], scaler=g_u_scaler, data_len=x_len)

    metrics_train = compute_metrics(outputs[0], outputs[1])
    metrics_test = compute_metrics(outputs[2], outputs[3])

    log.write(print_testing(metrics_train=metrics_train, metrics_test=metrics_test))

    g_u_pred = outputs[3].numpy()
    plot_rse_in_time(g_u=outputs[2], g_u_pred=g_u_pred, t_len=t_len, output_folder=output_folder, plot_name="rse_in_time")

    return outputs[2], g_u_pred