import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce