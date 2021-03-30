import torch
from models.neuralnet import NeuralNet
from models.convnet import ConvNet
import torch.nn as nn

if __name__ == '__main__':
    classifier = ConvNet(3, [['conv2d', '15,3,2'], ['conv2dt', '35,3,2'],['conv2d', '55,3,2'],['conv2dt', '50,3,2']],
                         [[nn.MaxPool2d((2,2)),nn.BatchNorm2d(15),nn.ReLU()],'skip', nn.ReLU(), nn.ReLU()])

    print(classifier(torch.randn(1,3,100,100)).shape)
