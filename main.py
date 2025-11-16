import argparse
import sys
from utils.modelfactory import ModelFactory 
from utils.datasetfactory import DatasetFactory 
from utils.context import TrainContext, TrainParams





def main():
   parser = argparse.ArgumentParser(
                prog='main',
                description='Neural Network environment in Python',
                epilog='Epilog Text')
   parser.add_argument('-m','--model',choices=['VGG_A'])
   parser.add_argument('-d','--dataset',choices=['ImageNet'])
   subparsers = parser.add_subparsers(dest='subcommand')
   parser_train = subparsers.add_parser('train',help='train help')
   parser_train.add_argument('-e','--epochs',default=10)
   parser_train.add_argument('-b','--batch_size',default=16)
   parser_train.add_argument('-s','--shuffle',action='store_true')
   parser_train.add_argument('-w','--num_workers',default=1)

   parser_test = subparsers.add_parser('test',help='test help')
   parser_test.add_argument('-t','--test')

   args = parser.parse_args()

   print("args:")
   print(args)

   mf = ModelFactory()
   df = DatasetFactory()
   model = mf.create(args.model)
   dataset = df.create(args.dataset,args.model)
   # TODO: Make params polymorph if possible
   if args.subcommand == "train":
       params = TrainParams
       #TODO: move parameters from args to params (using vars)
       ctx = TrainContext(params)
   ctx.set_model(model)
   ctx.set_dataset(dataset)
   ctx.set_train_params()
   ctx.train()

if __name__ == "__main__":
    sys.exit(main())
