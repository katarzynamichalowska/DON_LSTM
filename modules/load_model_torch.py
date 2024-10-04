import torch
import os
from modules.dir_functions import read_max_checkpoint

def load_fno_model(folder, model_name, device, cp=None):
    """ Load model in torch """
    try:
        checkpoint_dir = os.path.join(folder, 'checkpoints')
        if cp is None:
            cp = read_max_checkpoint(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt").format(epoch=(int(cp)))
        model = torch.load(os.path.join(folder, model_name), map_location=device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except:
        model = None
        print("Could not load model at {}".format(checkpoint_path))
    return model