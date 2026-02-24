import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: TestModel
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(3, 64, 3, stride=1, padding=0)
        self.dense1 = nn.Linear(57600, 10)

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        return x