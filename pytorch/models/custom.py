import torch.nn as nn
import torch


class CustomModel(nn.Module):
    def __init__(self, layers, output_function):
        super(CustomModel, self).__init__()
        for x in layers:
            x.train()
        self.model = nn.Sequential(*layers)
        self.preprocess = output_function

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.preprocess(x)
        return x
