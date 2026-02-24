import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: DropoutNet
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        return x