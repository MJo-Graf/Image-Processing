from datasets.ImageNet.dataset import ILSVRC
from torchvision import transforms

Transform = {
        "ImageNetVGG_A": transforms.Compose([transforms.Resize(size=[224,224]),
                                       transforms.Lambda(lambda img: img.convert('RGB')),
                                       transforms.ToTensor(),
                                       transforms.Lambda(lambda x: torch.cat((x,x,x),dim=0) if x.shape[0] == 1 else x),
                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        }

class DatasetFactory:
    def create(self,model: str,dataset: str):
        if model == "ImageNet":
            return ILSVRC(Transform[model+dataset])


