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
        self.dense1 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.dense1(x)
        return x

# Training: Net on data
def train_net():
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(data):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/10, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(data)
        print(f'Epoch {epoch+1}/10 completed. Average Loss: {avg_loss:.4f}')
    
    return model

if __name__ == "__main__":
    print('Starting training: Net on data')
    model = train_net()
    print('Training completed!')