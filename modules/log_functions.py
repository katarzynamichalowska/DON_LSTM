import io
import os
import numpy as np
import datetime
import pprint


def print_time(text="Time"):
    string = "{}: {}".format(text, datetime.datetime.now())

    return string


def pprint_layer_dict(layers):
    layers = pprint.pformat(layers, indent=1, compact=False, sort_dicts=False)
    layers = '\t'+'\t'.join(layers.splitlines(True))
    return layers


def print_params(module_name, table=False, header=False):
    items = dict([(item, getattr(module_name, item))
                 for item in dir(module_name) if not item.startswith("__")])
    string = ""
    if header:
        string += "\n\n#--------------------     PARAMETERS     --------------------#\n\n"
    if table:
        string += table_dictionary_items(items)
    else:
        string += equal_dictionary_items(items)

    print(string)

    return string


def table_dictionary_items(items):
    string = ""
    for key in items:
        string += "{:<30}:\t{:<50}\n".format(str(key), str(items[key]))
    return string


def equal_dictionary_items(items):
    string = ""
    for key in items:
        value = items[key]
        value = f"\"{value}\"" if isinstance(value, str) else value
        string += "{} = {}\n".format(str(key), str(value))
    return string


def print_model_summary(model, modelname):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    string = stream.getvalue()
    stream.close()
    header = "\n\n#--------------------        {}        --------------------#\n\n".format(
        modelname.upper())

    string = header + string
    print(string)

    return string

def print_activations(model, modelname):

    header = "\n\n#--------------------        {}        --------------------#\n\n".format(
        modelname.upper() + " ACTIVATIONS")

    string = header

    for i, layer in enumerate (model.layers):
        layer_name = str(i) + "\t" + str(layer).split(".")[-1].split(" ")[0]
        try:
            activation = str(layer.activation).split(" ")[1]
            string += "{:<20}:\t{:<20}\n".format(str(layer_name), str(activation))
        except AttributeError:
            string +=  "{:<20}:\t{:<20}\n".format(str(layer_name), "---")#f"{layer_name}\n"

    return string


def print_scale_info(u_train, u_test, g_u_train, g_u_test, xt_train, xt_test):

    def _min_range(variable):
        min_value = np.round(variable.min(axis=0).min(), 5)
        max_value = np.round(variable.min(axis=0).max(), 5)
        
        return min_value, max_value

    def _max_range(variable):
        min_value = np.round(variable.max(axis=0).min(), 5)
        max_value = np.round(variable.max(axis=0).max(), 5)

        return min_value, max_value
    
    def _string_ranges(name, variable):
        string  = name + ":\n"
        string += "\tmin: ({})\n".format(_min_range(variable))
        string += "\tmax: ({})\n".format(_max_range(variable))

        return string

    def _print_ranges(name, variable):
        string = ""
        if isinstance(variable, (list, tuple)):
            for i, v in enumerate(variable):
                n = f"{name}_{i}"
                string += _string_ranges(n, v)
        else:
             string += _string_ranges(name, variable)
        
        return string

    string = "\n\n#--------------------      SCALING      --------------------#\n\n"

    string += "Training data scale (min-max values range per column):\n\n"
    for name, variable in zip(["u", "g_u", "xt"], [u_train, g_u_train, xt_train]):
        string += _print_ranges(name, variable)

    string += "\n\nTesting data scale (min-max values range per column):\n\n"
    for name, variable in zip(["u", "g_u", "xt"], [u_test, g_u_test, xt_test]):
        string += _print_ranges(name, variable)

    return string


def print_training_history(history):
    string = "\n\n#--------------------      TRAINING      --------------------#\n\n"
    string += history

    return string


def print_testing(metrics_train, metrics_test, header=True, text=""):
    if header:
        string = "\n\n#--------------------     TESTING     --------------------#\n\n"
    else:
        string = ""


    if metrics_train and metrics_test:
        string += "MAE train: {},\ttest: {}\n".format(
            metrics_train['mae'], metrics_test['mae'])
        string += "RMSE train: {},\ttest: {}\n".format(
            metrics_train['rmse'], metrics_test['rmse'])
        string += "RSE train: {},\ttest: {}\n".format(
            metrics_train['rse'], metrics_test['rse'])
        string += "\n\n"
    
    elif metrics_test:
        string += f"Test {text}:\n"
        string += "MAE:\t{}\t".format(np.round(metrics_test['mae'],3))
        string += "RMSE:\t{}\t".format(np.round(metrics_test['rmse'],3))
        string += "RSE:\t{}\t".format(np.round(metrics_test['rse'],3))
        string += "\n"

    elif metrics_train:
        string += f"Train {text}:\n"
        string += "MAE:\t{}\t".format(np.round(metrics_train['mae'],3))
        string += "RMSE:\t{}\t".format(np.round(metrics_train['rmse'],3))
        string += "RSE:\t{}\t".format(np.round(metrics_train['rse'],3))
        string += "\n"

    print(string)

    return string


def parameter_from_logs(folder, param):
    with open(os.path.join(folder, "log.out"), "r") as fi:
        for ln in fi:
            if ln.startswith(param):
                param_value = ln[len(param):].split(
                    "\t")[1].split("\n")[0].strip(" ")

    return param_value
