import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class ASLCNN(nn.Module):
    def __init__(self, classes):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(138*25*25, 512)
        self.fc2 = nn.Linear(512, classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

