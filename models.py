import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ASLCNN(nn.Module):
    def __init__(self, classes):
        super(ASLCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Residual connections
        self.residual1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.residual2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.residual3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.residual4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Pooling, activation, and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        # Layer 1
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        identity = self.residual1(x)
        x = x + identity

        # Layer 2
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        identity = self.residual2(x)
        x = x + identity

        # Layer 3
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        identity = self.residual3(x)
        x = x + identity

        # Layer 4
        x = self.pool(self.leaky_relu(self.bn4(self.conv4(x))))
        identity = self.residual4(x)
        x = x + identity

        # Layer 5
        x = self.pool(self.leaky_relu(self.bn5(self.conv5(x))))

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)

        return x
