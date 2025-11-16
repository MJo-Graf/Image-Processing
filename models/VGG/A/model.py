import torch.nn as nn
import torch.nn.functional as F

class VGG_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=512*7*7,out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096,out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096,out_features=1000)
        )

    def forward(self,x):
        return self.layers.forward(x)




#import torch
#
#class VGG_A(nn.Module):
#    def __init__(self):
#        super(VGG_A,self).__init__()
#        super().__init__()
#        self.conv1 = nn.Conv2d(3,64,3,padding='same')
#        self.maxpooling = nn.MaxPool2d(2,stride=2)
#        #maxpool
#        self.conv2 = nn.Conv2d(64,128,3,padding='same')
#        #maxpool
#        self.conv3 = nn.Conv2d(128,256,3,padding='same')
#        self.conv4 = nn.Conv2d(256,256,3,padding='same')
#        #maxpool
#        self.conv5 = nn.Conv2d(256,512,3,padding='same')
#        self.conv6 = nn.Conv2d(512,512,3,padding='same')
#        #maxpool
#        self.conv7 = nn.Conv2d(512,512,3,padding='same')
#        self.conv8 = nn.Conv2d(512,512,3,padding='same')
#        #maxpool
#        self.fc1 = nn.Linear(512*7*7,4096)
#        self.fc2 = nn.Linear(4096,4096)
#        self.fc3 = nn.Linear(4096,1000)
#        #softmax
#        self.softmax = nn.Softmax(1)
#
#    def forward(self,x):
#        c1 = F.relu(self.conv1(x))
#        c2 = self.maxpooling(c1)
#        c3 = F.relu(self.conv2(c2))
#        c4 = self.maxpooling(c3)
#        c5 = F.relu(self.conv3(c4))
#        c6 = F.relu(self.conv4(c5))
#        c7 = self.maxpooling(c6)
#        c8 = F.relu(self.conv5(c7))
#        c9 = F.relu(self.conv6(c8))
#        c10 = self.maxpooling(c9)
#        c11 = F.relu(self.conv7(c10))
#        c12 = F.relu(self.conv8(c11))
#        c13 = self.maxpooling(c12)
#        c14 = torch.flatten(c13,start_dim=1)
#        c15 = F.relu(self.fc1(c14))
#        c16 = F.relu(self.fc2(c15))
#        c17 = self.fc3(c16)
#        return c17
#
#
