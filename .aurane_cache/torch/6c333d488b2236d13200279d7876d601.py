import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: ModelA
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.dense1(x)
        return x

# Model: ModelB
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 20)

    def forward(self, x):
        x = self.dense1(x)
        return x