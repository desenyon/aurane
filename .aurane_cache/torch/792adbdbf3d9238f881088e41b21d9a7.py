import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: DenseNet
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(784, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x