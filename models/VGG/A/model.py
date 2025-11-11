import torch.nn as nn
import torch.nn.functional as F

class VGG_A(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=64,out_channels=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=128,out_channels=256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,out_channels=256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=256,out_channels=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,out_channels=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=512,out_channels=512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,out_channels=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Linear(in_features=512*7*7,out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096,out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096,out_features=1000)
        )

    def forward(self,x):
        return nn.Sequential.forward(x)

