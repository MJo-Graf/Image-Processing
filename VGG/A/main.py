import kagglehub
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.optim as optim
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

class ILSVRCDataset(Dataset):
    def __init__(self,dataset_dir,transform=None,target_transform=None):
        self.dataset_dir = dataset_dir
        self.trainset_dir = dataset_dir + "/ILSVRC/Data/CLS-LOC/train"
        self.transform = transform
        p = Path(self.trainset_dir).resolve()
        self.imagelist = []
        self.synsetids = []
        self.descrlist = [] 
        with open(dataset_dir+"/LOC_synset_mapping.txt") as f:
            for l in f:
                self.synsetids.append(l.split()[0])
                self.descrlist.append(l.split()[1:])
        
        self.labellist = []
        for classdirs in sorted(p.iterdir()):
            for image in sorted(classdirs.iterdir()):
                self.imagelist.append(Path(image))
                image_synsetid = os.path.split(str(self.imagelist[-1].parent))[-1]
                self.labellist.append(self.synsetids.index(image_synsetid))
        

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self,idx):
        return self.transform(Image.open(self.imagelist[idx])),self.labellist[idx]

class VGG_A(nn.Module):
    def __init__(self):
        super(VGG_A,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding='same')
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
        self.softmax = nn.Softmax(1)

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
        c14 = torch.flatten(c13,start_dim=1)
        c15 = F.relu(self.fc1(c14))
        c16 = F.relu(self.fc2(c15))
        c17 = self.fc3(c16)
        return c17




def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Training on %s: " %(device))
    dataset_dir =  "/home/maximilian/data/ILSVRC"
    trainset=ILSVRCDataset(dataset_dir=dataset_dir,
            transform=transforms.Compose([transforms.Resize(size=[224,224]),
                                          transforms.Lambda(lambda img: img.convert('RGB')),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: torch.cat((x,x,x),dim=0) if x.shape[0] == 1 else x),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
              )

    net = VGG_A()
    net.to(device)
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=1)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start training, number of samples in training set = %d" %(trainset.__len__()))
    PATH = 'VGG11_net.pth'
    net.load_state_dict(torch.load(PATH,weights_only=True))

    for epoch in range(10):
        for i, data in enumerate(trainloader,0):
            inputs,labels= data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            oldweights = net.fc1.weight.clone()
            optimizer.step()
            print("[%f,%d] loss = %f" %(epoch,i,loss.item()))

    #PATH = 'VGG11_net.pth'
    #torch.save(net.state_dict(),PATH)
    #print("end of main")




if __name__ == "__main__":
    main()
