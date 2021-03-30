import torch.nn as nn
import math
import torch


class NeuralNet(nn.Module):
    """
    input_shape(int): Number of features for input. \n
    layer_size(list): Params for output size of layers can be [5,15,25,1] or [[5,True],[15,False]..,[1,True]]
    to modify if bias to be used, by default bias= True. \n
    post_layers(list): Post linear transformations can be \n
    [nn.Relu(),nn.Relu()], \n
    [[nn.Relu(),nn.BatchNorm1d(number of output of previous layer)],nn.Relu()] or \n
    [nn.Relu(),"skip",nn.Relu()] -> will skip transformation for layer3
    """

    def __init__(self, input_shape: int, layer_size: list, post_layers: list):
        super(NeuralNet, self).__init__()
        layers = []
        first = False
        last_layer = 0
        for x, y in zip(layer_size, post_layers):
            if type(x) is not list:
                if first is False:
                    layers.append(nn.Linear(input_shape, x))
                    last_layer = x
                    first = True
                else:
                    layers.append(nn.Linear(last_layer, x))
                    last_layer = x
            else:
                if first is False:
                    layers.append(nn.Linear(input_shape, x[0], x[1]))
                    last_layer = x[0]
                    first = True
                else:
                    layers.append(nn.Linear(last_layer, x[0], x[1]))
                    last_layer = x[0]
            if type(y) == list:
                for item in y:
                    layers.append(item)
            else:
                if y == "skip":
                    continue
                else:
                    layers.append(y)
        self.fc = nn.Sequential(*layers)
        self.fc = self.fc.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        :param m: Initialization of weights for Linear layers with 1/sqrt(input_shape) which are uniform distributed \n

        This stackoverflow answer will provide with why is initialized like this \n
        https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
        """
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1 / math.sqrt(n)
            torch.nn.init.uniform_(m.weight, -y, y)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
