import os
import numpy as np
import pandas as pd
from datetime import datetime
import params_default


def make_output_dir(params, is_testing=False):
    """
    Makes an output folder for the model.
    """
    
    if is_testing:
        output_folder = os.path.join(os.path.dirname(params.OUTPUT_FOLDER), "TEST")
    else:
        output_folder = params.OUTPUT_FOLDER + "_" + timestamp_now()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif is_testing:
        os.makedirs(output_folder, exist_ok=True)
    else:
        if not params.START_FROM_LATEST:
            output_folder = output_folder + "_copy"
            os.makedirs(output_folder, exist_ok=True)

    return output_folder

def timestamp_now():
    """
    Produces a timestamp of today's date and time.
    """
    timestamp = str(datetime.now())[2:22].replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    timestamp_now = timestamp[:6] + "_" + timestamp[6:]

    return timestamp_now

def timestamp_today():
    """
    Produces a timestamp of today's date.
    """
    timestamp = str(datetime.now())[2:22].replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    timestamp_date = timestamp[:6]

    return timestamp_date


def correct_dtype(val):
    """
    Corrects the data type of the value.
    """

    if val is not None:
        if _is_float(val):
        # Float
            val = float(val)
            if val-int(val)==0:
            # Int
                val = int(val)
        elif val=="True":
        # Bool
            val = True
        elif val=="False":
        # Bool
            val = False
        elif val=="None":
        # None
            val = None
        elif ("(" in val) and (")" in val):
        # Tuple
            val = val.strip("(").strip(")").split(",")
            val = tuple([correct_dtype(v) for v in val])
        elif isinstance(val, str):
        # String
            val = val.strip("\"")

    return val

def read_params_module_as_dict(module_name):
    """
    Reads all parameters from a module as a dictionary.
    """
    module = globals().get(module_name, None)
    if module:
        parameters = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    return parameters


def read_parameters(params_path):
    """
    Reads parameters to a dictionary from a params.txt file.
    """
    d = {}
    with open(params_path) as f:
        for line in f:
            keyval = line.split(" = ")
            key = keyval[0]
            val = keyval[1:]
            val = " = ".join(val)
            val = val.strip("\n")
            val = correct_dtype(val)
            d[key] = val
    return d

def _is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def _parameter_from_logs(modelname, param, folder="../models", logfile="log.out"):
    """
    Read a parameter from the logs of a model.
    """
    param_path = os.path.join(os.path.join(folder, modelname), logfile)

    with open(param_path,"r") as fi:
        for ln in fi:
            if ln.startswith(param):
                param_value = ln[len(param):].split("\t")[1].split("\n")[0].strip(" ")
                param_value = correct_dtype(param_value)
    return param_value
       

def read_model_params(params_to_read, modelname, folder="../models", logfile="log.out"):
    """
    Read parameters from the logs of a model.
    """
    params_vals= list()

    for p in params_to_read:
        try:
            params_vals.append(_parameter_from_logs(modelname=modelname, param=p.upper(), folder=folder, logfile=logfile))
        except:
            try:
                params_vals.append(_parameter_from_logs(modelname=modelname, param=p.lower(), folder=folder, logfile=logfile))
            except:
                try:
                    params_vals.append(getattr(params_default, p.upper()))
                except:
                    print(f"Couldn't find {p} for model {modelname}.")

    return params_vals

def read_max_checkpoint(checkpoint_dir):
    """
    Read the maximum checkpoint number from the checkpoint directory.
    Returns:
        @ cp_max: maximum checkpoint number (int)
    """
    cp_files = os.listdir(checkpoint_dir)
    cp_int = np.array([i.split("-")[1].split(".")[0] for i in cp_files if i!="checkpoint"])
    cp_max = np.unique(cp_int.astype('int32'))[-1]

    return cp_max
