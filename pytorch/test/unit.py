import unittest
import torch
from ..models.neuralnet import NeuralNet


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


def call_main():
    unittest.main()


if __name__ == '__main__':
    unittest.main()
