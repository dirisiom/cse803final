import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data import *
from models import *


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

data = fetch_data()

# Split dataset
train_size = int(0.7 * len(data))
val_size = int(.15 * len(data))
test_size = len(data) - (train_size + val_size)
train, val, test = random_split(data, [train_size, val_size, test_size])

# Data Loaders
batch_size = 32
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)

model = ASLCNN(len(data.classes)).to(device)

crit = nn.CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=.001)


def train_model(m, c, o, epochs=10):
    for e in range(epochs):
        m.train()
        running = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            o.zero_grad()
            out = m(inputs)
            loss = c(out, labels)
            loss.backward()
            o.step()
            running += loss
        print(f'Epoch {e}/{epochs}, Loss: {running/len(train_loader)}')
    print('Done training!')


def plot_results(m, loader):
    m.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = m(images)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        print(f'Accuracy: {acc}')
