import numpy as np
import torch
import torch.nn as nn


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        # raise Exception("Error!")
        return x

    return x.to(device)

 
def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, data_loader, return_input=False, return_target=False):
    """Forward data to model.

    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      output_dict: {'audio_name': (N,)
                    'clipwise_output': (N, classes_num), 
                    'framewise_output': (N, frames_num, classes_num), 
                    (optional) 'target': (N, classes_num), 
                    (optional) 'strong_target': (N, frames_num, classes_num)}
    """

    device = next(model.parameters()).device
    output_dict = {}
    
    # Evaluate on mini-batch
    for n, batch_data_dict in enumerate(data_loader):
        
        # Predict
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])
        append_to_dict(output_dict, 'clipwise_output', 
            batch_output['clipwise_output'].data.cpu().numpy())

        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output', 
                batch_output['framewise_output'].data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

            if 'strong_target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'strong_target', 
                    batch_data_dict['strong_target'])
                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def do_mixup(x, mixup_lambda):
    """Mixup data.

    Args:
      x: (N, ...)
      mixup_lambda: (N,)

    Return:
      out: (N, ...)
    """
    output = x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]
    output = output.transpose(0, -1)
    return output