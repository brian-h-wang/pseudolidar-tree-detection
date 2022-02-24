"""
Converts a model state dict from the format used by the PointRCNN repo to the Open3D format.
"""
import torch
from typing import Dict
from copy import copy

input_path = "trained_pointrcnn/rcnn_epoch_70.pth"

ckpt = torch.load(input_path)

# Process state dict keys
new_model_state_dict = {}
ckpt_model_state_dict: Dict[str, torch.Tensor] = ckpt['model_state']
for key in ckpt_model_state_dict.keys():
    new_key = key
    new_key = new_key.replace("backbone_net", "backbone")
    new_key = new_key.replace("rcnn_net", "rcnn")
    print("Converted key '%s' to '%s'" % (key, new_key))
    new_model_state_dict[new_key] = ckpt_model_state_dict[key]

output_dict = {
    'epoch' : ckpt['epoch'],
    'it': ckpt['it'],
    'model_state_dict': new_model_state_dict,
    'optimizer_state_dict': ckpt['optimizer_state']
}

output_path = "trained_pointrcnn/rcnn_epoch_70_modified.pth"
torch.save(output_dict, output_path)