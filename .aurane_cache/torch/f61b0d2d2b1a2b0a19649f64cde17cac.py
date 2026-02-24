import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Experiment: TestExp
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
device = torch.device("cuda")

# Model: Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.dense1(x)
        return x