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


#import xml.etree.ElementTree as ET

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
         label = torch.tensor(4,dtype=torch.long)
         #print("label in getitem:")
         #print(label.shape)
         #label = torch.zeros(1000)
         #label[5] =  1
        # tree = ET.parse(label)
        # root = tree.getroot()
        # for child in root:
        #     print(child.tag,child.attrib)
         

         if self.transform:
             image = self.transform(image)
         if self.target_transform: #TODO: What needs the target_transform to look like 
             label = self.target_transform(label)
         return image, label




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
        #c17 = F.relu(self.fc3(c16))
        c17 = self.fc3(c16)
        return c17
        #print("dim of c17:")
        #print(c17.shape)
        #c18 = self.softmax(c17)
        #return c18


def DownloadDataset():
    #return kagglehub.dataset_download("thbdh5765/ilsvrc2012")
    #return kagglehub.dataset_download("hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0")
    return kagglehub.dataset_download("huanghanchina/pascal-voc-2012")



def main():

    #a = torch.tensor([2.],requires_grad=True)
    #b = torch.tensor([6.],requires_grad=True)
    ##a = torch.tensor([2.,3.],requires_grad=True)
    ##b = torch.tensor([6.,4.],requires_grad=True)
    #Q = 3*a**3 - b**2
    #Q.backward()
    #print(a.grad)
    #print(Q.grad_fn)
    #print(Q.grad_fn.next_functions)




    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Training on %s: " %(device))
    dataset_dir =  "/home/maximilian/data/ILSVRC"
    trainset=ILSVRCDataset(dataset_dir=dataset_dir,
            transform=transforms.Compose([transforms.Resize(size=[224,224]),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: torch.cat((x,x,x),dim=0) if x.shape[0] == 1 else x),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
              )
    #print("len of dataset %d=" %(trainset.__len__()))
    #image,label = trainset.__getitem__(229453)
    #print("image:")
    #print(image)
    #print(image.shape)
    #print("label: ")
    #print(label)
    #print(trainset.descrlist[label])
    #plt.imshow(image.permute(1,2,0))
    #plt.show()

    net = VGG_A()
    net.to(device)
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=1)
    criterion = torch.nn.CrossEntropyLoss()
    #test=torch.zeros(1,1000)
    #test[0,4]=52
    #tl = torch.tensor([42])
    #loss=criterion(test,tl)
    #print("LOSS ITEM=")
    #print(loss.item())
    #print("outputs dim=")
    #print(test.shape)
    # TODO: Fix

    print("Start training, number of samples in training set = %d" %(trainset.__len__()))

    for epoch in range(10):
        for i, data in enumerate(trainloader,0):
            inputs,labels= data[0].to(device),data[1].to(device)
            print("labels=")
            print(labels)
            #print("labels.requires_grad:")
            #print(labels.requires_grad)
            #labels.requires_grad =True
            #print(trainset.descrlist[labels[0]])
            #print("output dims=")
            #print(outputs.shape)
            #print("inputs[0] dims=")
            #print(inputs[0].shape)
            #plt.imshow(inputs[0].permute(1,2,0))
            #plt.show()
            optimizer.zero_grad()
            outputs=net(inputs)
           # value1=F.cross_entropy(outputs[0],labels[0])
           # nom = torch.exp(outputs[0,labels[0]])
           # denom = torch.exp(outputs[0]).sum()
           # ce= -torch.log(torch.div(nom,denom))
           # print("nom=")
           # print(torch.exp(outputs[0,labels[0]]))
           # print("denom=")
           # print(torch.exp(outputs[0]).sum())
           # print("labels[0]=")
           # print(labels[0])
           # print("value1=")
           # print(value1)
           # print("ce")
           # print(ce)
            #print("outputs.requires_grad:")
            #print(outputs.requires_grad)
            #print("outputs=")
            #print(outputs)
            #print("labels=")
            #print(labels)
            #labels=torch.tensor([42])
            loss = criterion(outputs,labels)
            loss.backward()
            #grads_exist = False 
            #for name, param in net.named_parameters():
            #    if param.grad is not None and param.grad.sum() != 0:
            #        print(param.grad)
            #        grads_exist = True
            #        break
            #if not grads_exist:
            #    print(f"ERROR: No gradients found or all gradients zero")
            #print("net parameters:")
            #print(net.parameters())
            #for param in net.parameters():
            #    print(param.grad)
            #    break
            #print("Loss Grad:")
            #print(loss.grad)
            oldweights = net.fc1.weight.clone()
            optimizer.step()
            #changed = not torch.equal(oldweights,net.fc1.weight)
            #if changed:
            #    print(oldweights)
            #    print("Weights have changed")
            print("[%f,%d] loss = %f" %(epoch,i,loss.item()))

    PATH = 'VGG11_net.pth'
    torch.save(net.state_dict(),PATH)
    print("end of main")




if __name__ == "__main__":
    main()
