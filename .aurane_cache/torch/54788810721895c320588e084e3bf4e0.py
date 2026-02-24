import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Model: ActivationNet
class ActivationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 16)
        self.dense4 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        x = F.softmax(self.dense4(x), dim=-1)
        return x