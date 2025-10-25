import kagglehub
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt


class VocDataset(Dataset):
    def __init__(self,dataset_dir,transform=None,target_transform=None):
        self.imgdir = dataset_dir + "/VOC2012/JPEGImages/"
        self.labelsdir = dataset_dir + "/VOC2012/Annotations/"
        self.imglist = sorted(os.listdir(self.imgdir))
        self.labellist = sorted(os.listdir(self.labelsdir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self,idx):
         image = Image.open(self.imgdir+self.imglist[idx])
         label = self.labelsdir+self.labellist[idx]
         if self.transform:
             image = self.transform(image)
         if self.target_transform: #TODO: What needs the target_transform to look like 
             label = self.target_transform(label)
         return image, label




class VGG_A(nn.Module):
    def __init__(self):
        super(VGG_A,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,padding='same')
        self.maxpooling = nn.MaxPool2d(2,stride=2)
        #maxpool
        self.conv2 = nn.Conv2d(64,128,3,padding='same')
        #maxpool
        self.conv3 = nn.Conv2d(128,256,3,padding='same')
        self.conv4 = nn.Conv2d(256,256,3,padding='same')
        #maxpool
        self.conv5 = nn.Conv2d(256,512,3,padding='same')
        self.conv6 = nn.Conv2d(512,512,3,padding='same')
        #maxpool
        self.conv7 = nn.Conv2d(512,512,3,padding='same')
        self.conv8 = nn.Conv2d(512,512,3,padding='same')

        #maxpool
        self.fc1 = nn.Linear(512*7*7,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        #softmax
        self.softmax = nn.Softmax(1000)

    def forward(self,x):
        c1 = F.relu(self.conv1(x))  
        c2 = self.maxpooling(c1)
        c3 = F.relu(self.conv2(c2))
        c4 = self.maxpooling(c3)
        c5 = F.relu(self.conv3(c4))
        c6 = F.relu(self.conv4(c5))
        c7 = self.maxpooling(c6)
        c8 = F.relu(self.conv5(c7))
        c9 = F.relu(self.conv6(c8))
        c10 = self.maxpooling(c9)
        c11 = F.relu(self.conv7(c10))
        c12 = F.relu(self.conv8(c11))
        c13 = self.maxpooling(c12)
        c14 = torch.flatten(c13,1)
        c15 = F.relu(self.fc1(c14))
        c16 = F.relu(self.fc2(c15))
        c17 = F.relu(self.fc3(c16))
        return c17


def DownloadDataset():
    return kagglehub.dataset_download("huanghanchina/pascal-voc-2012")



def main():

    print("Starting ...")
    dataset_dir =  DownloadDataset()
    dataset=VocDataset(dataset_dir=dataset_dir,
               transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=[224,224])]),
              )
    
    #image,label = dataset.__getitem__(1)
    #print("image.shape=")
    #print(image.numel())

    # TODO: Add transform for labels
    # TODO: Add training
    # TODO: Save model




if __name__ == "__main__":
    main()
