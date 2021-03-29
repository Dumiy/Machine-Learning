import torch.nn as nn
import math
import torch


class NeuralNet(nn.Module):
    def __init__(self, input_shape: int, layer_size: list, post_layers: list):
        super(NeuralNet, self).__init__()
        layers = []
        first = False
        last_layer = 0
        for x, y in zip(layer_size, post_layers):
            if x == layer_size[0] and first is False:
                layers.append(nn.Linear(input_shape, x))
                last_layer = x
                first = True
            else:
                layers.append(nn.Linear(last_layer, x))
                last_layer = x
            if type(y) == list:
                for item in y:
                    layers.append(item)
            else:
                layers.append(y)
        self.fc = nn.Sequential(*layers)
        self.fc = self.fc.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1 / math.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
