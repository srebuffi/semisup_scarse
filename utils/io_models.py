import os
import torch
import shutil


def save_checkpoint(state, is_best, save_path, checkpoint='checkpoint.pth', best_model='model_best.pth'):
    """ Save model. """
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, save_path + '/' + checkpoint)
    if is_best:
        shutil.copyfile(save_path + '/' + checkpoint, save_path + '/' + best_model)
