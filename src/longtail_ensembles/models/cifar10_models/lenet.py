"""
Implementation of Lenet5
Edited from
https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb"""
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "lenet5",
]


class LeNet5(nn.Module):

    def __init__(self, num_classes=10, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5), nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(), nn.MaxPool2d(kernel_size=2))

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        #probas = F.softmax(logits, dim=1)
        #return logits, probas
        return logits


def lenet5(pretrained=False, progress=True, device="cpu", **kwargs):
    r"""Lenet5 model architecture
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
      progress (bool): If True, displays a progress bar of the download to stderr
      aux_logits (bool): If True, adds two auxiliary branches that can improve training.
          Default: *False* when pretrained is True otherwise *True*
      transform_input (bool): If True, preprocesses the input according to the method with which it
          was trained on ImageNet. Default: *False*
  """
    model = LeNet5(**kwargs)
    if pretrained:
        raise NotImplementedError('pretrained model is NA')
    return model
