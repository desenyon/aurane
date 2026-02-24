import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: PoolNet
class PoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.conv2d2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2d2(x))
        x = F.avg_pool2d(x, 2, stride=2)
        return x