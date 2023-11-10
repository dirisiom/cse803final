import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Using {device}')