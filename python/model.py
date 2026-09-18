"""Canonical LeNet model used by the Route B baseline.

This file is deliberately independent from ``download.py``: it only defines the
network and the layer metadata that is shared with the HLS implementation.
"""
from collections import OrderedDict

import torch
from torch import nn


NETWORK_NAME = "LeNet5-MNIST"
NUM_CLASSES = 10


class LeNet5(nn.Module):
    """LeNet-5 adapted to 28x28, single-channel MNIST images.

    Bias terms are disabled to match the frozen hardware-oriented network.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, NUM_CLASSES, bias=False)

    def forward_with_intermediates(self, x):
        out = OrderedDict()
        out["input"] = x
        x = self.conv1(x)
        out["conv1"] = x
        x = torch.relu(x)
        out["relu1"] = x
        x = self.pool1(x)
        out["pool1"] = x
        x = self.conv2(x)
        out["conv2"] = x
        x = torch.relu(x)
        out["relu2"] = x
        x = self.pool2(x)
        out["pool2"] = x
        x = torch.flatten(x, start_dim=1)
        out["flatten"] = x
        x = self.fc1(x)
        out["fc1"] = x
        x = torch.relu(x)
        out["relu3"] = x
        x = self.fc2(x)
        out["fc2"] = x
        x = torch.relu(x)
        out["relu4"] = x
        x = self.fc3(x)
        out["logits"] = x
        return out

    def forward(self, x):
        return self.forward_with_intermediates(x)["logits"]


LAYER_TABLE = [
    {"name": "input", "input_shape": [1, 28, 28], "output_shape": [1, 28, 28], "kernel": None, "stride": None, "padding": 0},
    {"name": "conv1", "input_shape": [1, 28, 28], "output_shape": [6, 24, 24], "kernel": 5, "stride": 1, "padding": 0},
    {"name": "relu1", "input_shape": [6, 24, 24], "output_shape": [6, 24, 24], "kernel": None, "stride": None, "padding": 0},
    {"name": "pool1", "input_shape": [6, 24, 24], "output_shape": [6, 12, 12], "kernel": 2, "stride": 2, "padding": 0},
    {"name": "conv2", "input_shape": [6, 12, 12], "output_shape": [16, 8, 8], "kernel": 5, "stride": 1, "padding": 0},
    {"name": "relu2", "input_shape": [16, 8, 8], "output_shape": [16, 8, 8], "kernel": None, "stride": None, "padding": 0},
    {"name": "pool2", "input_shape": [16, 8, 8], "output_shape": [16, 4, 4], "kernel": 2, "stride": 2, "padding": 0},
    {"name": "flatten", "input_shape": [16, 4, 4], "output_shape": [256], "kernel": None, "stride": None, "padding": 0},
    {"name": "fc1", "input_shape": [256], "output_shape": [120], "kernel": None, "stride": None, "padding": 0},
    {"name": "relu3", "input_shape": [120], "output_shape": [120], "kernel": None, "stride": None, "padding": 0},
    {"name": "fc2", "input_shape": [120], "output_shape": [84], "kernel": None, "stride": None, "padding": 0},
    {"name": "relu4", "input_shape": [84], "output_shape": [84], "kernel": None, "stride": None, "padding": 0},
    {"name": "fc3", "input_shape": [84], "output_shape": [10], "kernel": None, "stride": None, "padding": 0},
]


def model_metadata():
    return {
        "network": NETWORK_NAME,
        "input_layout": "CHW",
        "input_shape": [1, 28, 28],
        "input_range": [0.0, 1.0],
        "weight_layout": "[out_channel][in_channel][kernel_h][kernel_w] for Conv; [out][in] for FC",
        "flatten_order": "C-contiguous CHW: c * H * W + h * W + w",
        "activation": "ReLU after each Conv and first two FC layers",
        "pooling": "2x2 non-overlapping max pooling, stride 2",
        "bias": False,
        "num_classes": NUM_CLASSES,
        "layers": LAYER_TABLE,
    }
