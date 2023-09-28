"""
Loading DeepONets, FNOs and RNNs.
"""
import sys
sys.path.insert(0, "../")
from modules.dir_functions import read_max_checkpoint
import os
import json
import tensorflow as tf
import numpy as np


def load_model(main_folder, modelname="model3", checkpoint=10000):
    """ Load model in tensorflow """
    if checkpoint is None:
        model, _ = load_model_latest(modelname=modelname, directory=main_folder)
    
    else:
        model_architecture = "{}/model_structure.json".format(modelname)
        model_architecture = os.path.join(main_folder, model_architecture)

        with open(model_architecture, 'r') as json_file:
            architecture = json.load(json_file)
            model = tf.keras.models.model_from_json(json.dumps(architecture))

        weights_path = "{model}/checkpoints/cp-{cp:04d}.ckpt".format(model=modelname, cp=checkpoint)
        weights_path = os.path.join(main_folder, weights_path)
        try:
            model.load_weights(weights_path).expect_partial()
            return model
        except tf.errors.NotFoundError:
            print("{model}/checkpoints/cp-{cp:04d}.ckpt not found.".format(model=modelname, cp=checkpoint))
            return None
    

def load_model_latest(modelname, directory="../models"):
    """
    Load the latest model from a directory (the one with the highest checkpoint number).
    """
    checkpoint_dir = os.path.join(directory, f"{modelname}/checkpoints")
    cp_max = read_max_checkpoint(checkpoint_dir)
    model = load_model(directory, modelname=modelname, checkpoint=cp_max)
    return model, cp_max