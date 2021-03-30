import torch
from .utils import *


class ConvNet(nn.Module):
    def __init__(self, input_channels: int, layers_params: list, post_layers: list):
        """

        :param input_channels: number of channels for input data
        :param layers_params: list of [type of convolution,'convolutions params']
        :param post_layers: Post linear transformations can be \n
        [nn.Relu(),nn.Relu()], \n
        [[nn.Relu(),nn.BatchNorm1d(number of output of previous layer)],nn.Relu()] or \n
        [nn.Relu(),"skip",nn.Relu()] -> will skip transformation for layer3
        """
        super(ConvNet, self).__init__()
        self.conv_dict = return_conv_dict()
        layers = []
        first = False
        last_layer = 0
        for x, y in zip(layers_params, post_layers):
            if first is False:
                if 'out_channels' in x:
                    x[1] = "in_channels=" + str(input_channels) + "," + x[1]
                    layers.append(self.create_conv(x))
                else:
                    x[1] = str(input_channels) + "," + x[1]
                    layers.append(self.create_conv(x))
                last_layer = layers[-1].out_channels
                first = True
            else:
                if 'out_channels' in x:
                    x[1] = "in_channels=" + str(last_layer) + "," + x[1]
                    layers.append(self.create_conv(x))
                else:
                    x[1] = str(last_layer) + "," + x[1]
                    layers.append(self.create_conv(x))
                last_layer = layers[-1].out_channels
            if type(y) == list:
                for item in y:
                    layers.append(item)
            else:
                if y == "skip":
                    continue
                else:
                    layers.append(y)
        self.convnet = nn.Sequential(*layers)
        self.convnet = self.convnet.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) in return_conv_dict_values():
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def create_conv(self, params: list) -> torch.nn:
        """

        params contains an list of 2 items \n First item is type of convolution, which can be any of \n
        ['conv1d','conv2d','conv3d','conv1dt','conv2dt','conv3dt'], where t is short for transpose \n
        Second item is a string of params for convolution with specific arguments \n
        'out_channels=3,kernel_size=3' or '3,3' with the order of convolution base params \n

        :param params(list) ['conv_name','parameters for convolution']
        :return: Convolutional type layer
        """
        convolution = eval(str(self.conv_dict[params[0].lower()])[8:-2] + "(" + params[1] + ")")
        return convolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnet(x)
