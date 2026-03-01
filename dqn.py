import torch
import torch.nn as nn
from torchsummary import summary

class DQN(nn.Module):
    def __init__(self, in_classes, out_classes):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
model = DQN(8, 10)
summary(model, (8,))