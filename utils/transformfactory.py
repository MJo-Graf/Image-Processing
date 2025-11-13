from torchvision import transforms

class TransformFactory:
    def create(self,model,dataset):
        if model == "VGG_A" and dataset == "ILSVRC":
            return transforms.Compose([transforms.Resize(size=[224,224]),
                                       transforms.Lambda(lambda img: img.convert('RGB')),
                                       transforms.ToTensor(),
                                       transforms.Lambda(lambda x: torch.cat((x,x,x),dim=0) if x.shape[0] == 1 else x),
                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

