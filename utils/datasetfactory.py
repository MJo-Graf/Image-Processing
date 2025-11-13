from datasets.ImageNet.dataset import ILSVRC
class DatasetFactory:
    def create(self,name):
        if name == "ImageNet":
            return ILSVRC()
