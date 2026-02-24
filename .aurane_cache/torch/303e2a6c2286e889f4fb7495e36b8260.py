import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Dataset: my_data
my_data_dataset = torchvision.datasets.MNIST(transform=transforms.ToTensor(), download=True)
my_data = DataLoader(my_data_dataset, batch_size=32, shuffle=True)

# Model: Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.dense1(x)
        return x