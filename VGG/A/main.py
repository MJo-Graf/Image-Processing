import kagglehub
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset

class VocDataset(Dataset):
    def __init__(self,path):
        self.imgdir = path + "/VOC2012/JPEGImages/"
        self.labelsdir = path + "/VOC2012/Annotations/"
        self.imglist = sorted(os.listdir(self.imgdir))
        self.labellist = sorted(os.listdir(self.labelsdir))

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self,idx):
         image = self.imgdir+self.imglist[idx]
         label = self.labelsdir+self.labellist[idx]
         return image, label




class VGG_A(nn.Module):
    def __init__(self):
        super(VGG_A,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3)
        self.maxpooling = nn.MaxPool2d(2,stride=2)
        #maxpool
        self.conv2 = nn.Conv2d(64,128,3)
        #maxpool
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv4 = nn.Conv2d(256,256,3)
        #maxpool
        self.conv5 = nn.Conv2d(256,512,3)
        self.conv6 = nn.Conv2d(512,512,3)
        #maxpool
        self.conv7 = nn.Conv2d(512,512,3)
        self.conv8 = nn.Conv2d(512,512,3)
        #maxpool
        self.fc1 = nn.Linear(512,4096)
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
        c14 = F.relu(self.fc1(c13))
        c15 = F.relu(self.fc1(c14))
        c16 = F.relu(self.fc1(c15))
        return c16

        

        








def DownloadDataset():
    return kagglehub.dataset_download("huanghanchina/pascal-voc-2012")



def main():
    print("Starting ...")
    path_dataset =  DownloadDataset()
    net = VGG_A()
    print(path_dataset)
    dataset = VocDataset(path_dataset)
    print("number of jpeg files= %s" % (dataset.__len__()))
    #print(os.listdir(path_dataset))
    image,label = dataset.__getitem__(2)
    print("image = %s, label = %s" %(image,label))



if __name__ == "__main__":
    main()
