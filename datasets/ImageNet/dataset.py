import kagglehub
import os
from pathlib import Path
from torch.utils.data import Dataset



class ILSVRC(Dataset):
    def __init__(self,transform=None,target_transform=None):
        dataset_dir = kagglehub.competition_download(handle='imagenet-object-localization-challenge')
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


