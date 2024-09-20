"""
Loading DeepONets, FNOs and RNNs.
"""
import sys
sys.path.insert(0, "../")
import os
import json
import tensorflow as tf

def load_model(main_folder, checkpoint):
    """ Load model in tensorflow """
    model_architecture = os.path.join(main_folder, "model_structure.json")

    with open(model_architecture, 'r') as json_file:
        architecture = json.load(json_file)
        model = tf.keras.models.model_from_json(json.dumps(architecture))

    weights_path = "cp-{cp:04d}.ckpt".format(cp=checkpoint)
    weights_path = os.path.join(main_folder, "checkpoints", weights_path)
    try:
        model.load_weights(weights_path).expect_partial()
        return model
    except tf.errors.NotFoundError:
        print("{ckpt} not found.".format(weights_path))
        return None
    