import torch
import os
from modules.dir_functions import read_max_checkpoint

def load_fno_model(folder, cp=None):
    """ Load model in torch """
    try:
        model = torch.load(folder)
        checkpoint_dir = os.path.join(folder, 'checkpoints')
        if cp is None:
            cp = read_max_checkpoint(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")
        model.load_state_dict(torch.load(checkpoint_path.format(epoch=(int(cp)))))
    except:
        model = None
        print("Could not load model at {}".format(folder))

    return model