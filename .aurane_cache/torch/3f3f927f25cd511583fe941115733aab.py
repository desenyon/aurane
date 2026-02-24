import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 32, 3, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        return x