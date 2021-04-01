import unittest
import torch
from ..models.neuralnet import NeuralNet
from ..models.convnet import ConvNet
import torch.nn as nn


class Class(unittest.TestCase):
    def test_simple_neural_network(self):
        model = NeuralNet(100, [50, 60, 80], [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()])
        model.eval()
        self.assertIsInstance(model, NeuralNet, "Should create a NeuralNet class")

    def test_complex_neural_network(self):
        model = NeuralNet(5, [15, 20, 2],
                          [[torch.nn.ReLU(), torch.nn.BatchNorm1d(15)], [torch.nn.ReLU(), torch.nn.BatchNorm1d(20)],
                           torch.nn.Sigmoid()])
        model.eval()
        self.assertIsInstance(model, NeuralNet, "Should create a NeuralNet class")

    def test_run_simple_neural_network(self):
        model = NeuralNet(100, [50, 60, 80], [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()])
        model.eval()
        var = torch.randn(1, 100, requires_grad=False)
        self.assertEqual(model(var).shape, (1, 80),
                         "Should have the same output shape with (1,80), but got {} instead".format(model(var).shape))

    def test_run_complex_neural_network(self):
        model = NeuralNet(5, [15, 20, 2],
                          [[torch.nn.ReLU(), torch.nn.BatchNorm1d(15)], [torch.nn.ReLU(), torch.nn.BatchNorm1d(20)],
                           torch.nn.Sigmoid()])
        model.eval()
        var = torch.randn(1, 5, requires_grad=False)
        self.assertEqual(model(var).shape, (1, 2),
                         "Should have the same output shape with (1,2), but got {} instead".format(model(var).shape))

    def test_simple_convnet(self):
        model = ConvNet(3, [['conv2d', '15,3,2'],
                            ['conv2dt', '35,3,2'],
                            ['conv2dt', '55,3,2'],
                            ['conv2d', '50,3,2']],
                        [[nn.MaxPool2d((2, 2)), nn.BatchNorm2d(15), nn.ReLU()],
                         'skip', nn.ReLU(),
                         nn.ReLU()])
        model.eval()
        self.assertIsInstance(model, ConvNet, "Should create a ConvNet class")

    def test_run_simple_convnet(self):
        model = ConvNet(3, [['conv2d', '15,3,2'],
                            ['conv2dt', '35,3,2'],
                            ['conv2dt', '55,3,2'],
                            ['conv2d', '50,3,2']],
                        [[nn.MaxPool2d((2, 2)), nn.BatchNorm2d(15), nn.ReLU()],
                         'skip', nn.ReLU(),
                         nn.ReLU()])
        model.eval()
        var = torch.randn(1, 3, 100, 100, requires_grad=False)
        self.assertEqual(model(var).shape, (1, 50, 49, 49),
                         "Should have the same output shape with (1,55,51,51), but got {} instead".format(
                             model(var).shape))

    def test_simple_params_convnet(self):
        model = ConvNet(3,
                        [['conv2d', 'out_channels=15,kernel_size=3,stride=2'],
                         ['conv2dt', 'out_channels=30,kernel_size=3,stride=2'],
                         ['conv2dt', 'out_channels=50,kernel_size=3,stride=2'],
                         ['conv2d', 'out_channels=55,kernel_size=3,stride=2,padding=(2,2)']],
                        [[nn.MaxPool2d((2, 2)), nn.BatchNorm2d(15), nn.ReLU()],
                         'skip',
                         nn.ReLU(),
                         nn.ReLU()])
        model.eval()
        self.assertIsInstance(model, ConvNet, "Should create a ConvNet class")

    def test_run_simple_params_convnet(self):
        model = ConvNet(3,
                        [['conv2d', 'out_channels=15,kernel_size=3,stride=2'],
                         ['conv2dt', 'out_channels=30,kernel_size=3,stride=2'],
                         ['conv2dt', 'out_channels=50,kernel_size=3,stride=2'],
                         ['conv2d', 'out_channels=55,kernel_size=3,stride=2,padding=(2,2)']],
                        [[nn.MaxPool2d((2, 2)), nn.BatchNorm2d(15), nn.ReLU()],
                         'skip',
                         nn.ReLU(),
                         nn.ReLU()])
        model.eval()
        var = torch.randn(1, 3, 100, 100, requires_grad=False)
        self.assertEqual(model(var).shape, (1, 55, 51, 51),
                         "Should have the same output shape with (1,55,51,51), but got {} instead".format(
                             model(var).shape))


def call_main():
    unittest.main()


if __name__ == '__main__':
    unittest.main()
