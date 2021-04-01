import torch.nn as nn
from torch.utils.data import Dataset
import torch

import cv2 as cv


def return_conv_dict() -> dict:
    conv_dict = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'conv1dt': nn.ConvTranspose1d,
                 'conv2dt': nn.ConvTranspose2d, 'conv3dt': nn.ConvTranspose3d}
    return conv_dict


def return_conv_dict_keys() -> dict.keys:
    return return_conv_dict().keys()


def return_conv_dict_values() -> dict.values:
    return return_conv_dict().values()


class DataClass(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        :param data: data list with features
        :param labels: label list
        :param transform: Applying preprocessing on the data
        """
        self.df = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> (torch.tensor, torch.tensor):
        sample = torch.tensor(self.df[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class ImageClass(Dataset):
    def __init__(self, data=None, labels=None, transform=None, base_path="", greyscale=None):
        """
        :param data: data list with features
        :param labels: label list
        :param transform: Applying preprocessing on the data
        :param greyscale: Image to be read greyscale
        :param base_path: root dir to files ../folder/images

        """

        if labels is None:
            labels = []
        if data is None:
            data = []
        self.df = data
        self.labels = labels
        self.transform = transform
        self.greyscale = greyscale
        self.base_path = base_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> (torch.tensor, torch.tensor):
        if self.greyscale:
            sample = cv.imread(self.base_path + "/" + self.df[idx], 0)
        else:
            sample = cv.imread(self.base_path + "/" + self.df[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def train(epoch: int, network: nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader, loss: nn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.train()
    for data, target in train_loader:
        optimizer.zero_grad()

        output = network(data.to(device))

        loss_value = loss(output, target.to(device))

        loss_value.backward()

        optimizer.step()

    print('Train Epoch: {} Length {} \tLoss: {:.6f}'.format(epoch, len(train_loader), loss_value.item()))


def test_accuracy(epoch: int, network: nn.Module, test_loader: torch.utils.data.DataLoader, loss: nn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            target = target.to(device)
            data = data.to(device)

            outputs = network(data)

            test_loss = loss(outputs, target)

            correct += (outputs == target).sum()

    print('Test Epoch: {} Length {} \tLoss: {:.6f}'.format(epoch, len(test_loader), test_loss.item()))
    print("Accuracy - ", (correct / len(test_loader)) * 100)
