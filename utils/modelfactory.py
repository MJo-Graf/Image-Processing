from models.VGG.A.model import VGG_A

class ModelFactory:
    def create(self,name):
        if name == "VGG_A":
            return VGG_A()

