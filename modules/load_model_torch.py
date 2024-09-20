import torch
import os
from modules.dir_functions import read_max_checkpoint

def load_fno_model(folder, modelname, cp=None):
    """ Load model in torch """
    path_model = os.path.join(folder, modelname)
    try:
        model = torch.load(path_model)
        checkpoint_dir = os.path.join(folder, 'checkpoints')
        if cp is None:
            cp = read_max_checkpoint(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
        model.load_state_dict(torch.load(checkpoint_path.format(epoch=(int(cp)))))
    except:
        model = None
        print("Could not load model {}".format(modelname))

    return model