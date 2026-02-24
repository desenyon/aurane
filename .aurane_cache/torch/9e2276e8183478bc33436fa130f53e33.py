import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.dense1 = nn.Linear(4096, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        return x