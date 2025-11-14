import torch

class Context:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = None

    def set_model(self,model):
        self.model = model 
        self.model.to(self.device)

    def set_dataset(self,dataset):
        self.dataset = dataset



TrainParams = {
        "epochs":10,
        "batch_size":64,
        "shuffle":True,
        "num_workers":1
        }

class TrainContext(Context):
    def __init__(self):
        super().__init__()
        self.criterion = None
        self.optimizer = None
        self.epochs = None

    def set_train_params(self,params=TrainParams):
        self.trainloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=params["batch_size"],
                shuffle=params["shuffle"],
                num_workers=params["num_workers"]
                )
        self.criterion = torch.nn.CrossEntropyLoss()
        print("self model paramters=")
        print(self.model.parameters())
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
        self.epochs = params["epochs"]


    def train(self):
        print("Start Training")

        for epoch in range(self.epochs):
            for i,data in enumerate(self.trainloader,0):
                inputs,labels = data[0].to(self.device),data[1].to(self.device)
                print("input shape: ")
                print(inputs.shape)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                print("[%d,%d] loss = %f"%(epoch,i,loss.item()))
