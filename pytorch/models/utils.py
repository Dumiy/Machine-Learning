import torch.nn as nn


def return_conv_dict() -> dict:
    conv_dict = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'conv1dt': nn.ConvTranspose1d,
                 'conv2dt': nn.ConvTranspose2d, 'conv3dt': nn.ConvTranspose3d}
    return conv_dict


def return_conv_dict_keys() -> dict.keys:
    return return_conv_dict().keys()


def return_conv_dict_values() -> dict.values:
    return return_conv_dict().values()
