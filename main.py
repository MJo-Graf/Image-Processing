from utils.modelfactory import ModelFactory 
from utils.datasetfactory import DatasetFactory 
from utils.transformfactory import TransformFactory 

def main():
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   mf = ModelFactory()
   tf = TransformFactory()
   model = mf.create("VGG_A")
   df = DatasetFactory()
   dataset = df.create("ImageNet")
   print(model)
   print(dataset)

if __name__ == "__main__":
    main()
