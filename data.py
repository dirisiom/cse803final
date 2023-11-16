import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.CenterCrop((195, 195)),  # remove the blue border
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def fetch_data():
    path = './data/asl_images'
    dset = datasets.ImageFolder(root=path, transform=transform)
    return dset
