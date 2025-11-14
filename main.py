import sys
from utils.modelfactory import ModelFactory 
from utils.datasetfactory import DatasetFactory 
from utils.context import TrainContext





def main():

   mf = ModelFactory()
   df = DatasetFactory()
   model = mf.create("VGG_A")
   dataset = df.create("ImageNet","VGG_A")
   ctx = TrainContext()
   ctx.set_model(model)
   ctx.set_dataset(dataset)
   ctx.set_train_params()
   ctx.train()
   print(model)
   print(dataset)

if __name__ == "__main__":
    sys.exit(main())
