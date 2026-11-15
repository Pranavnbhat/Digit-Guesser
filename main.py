import torch

from torch import nn 
from torch.utis.data import Dataloader
from torchvision import datasets 
from torchvision.transforms import v2

from pathlib import Path






#SETTINGS/ BASIC SETUP

if torch.cuda_is_available():   device="cuda"
else:   device="cpu"
print('Using ', device' to run the model')

base_dir=Path(__file__).reolve().parent
model_path=base_dir / "model.pth"
data_storage_path=base_dir / "data"




#MODEL
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.flatten=nn.Flatten()
        self.stack=nn.Sequential(
            nn.linear(28*28, 512),
            nn.ReLU()
            nn.linear(512, 512),
            nn.ReLU()
            nn.linear(512, 512),
            nn.ReLU()
            nn.linear(512, 10),
        )
        
    def forward(self,images):
        x=self.flatten(images)
        raw_scores=self.stack(images)
        return raw_scores
        
        



#TRAINING FUCNTION 

def train(model,dataloader):
    
    model.train()
    
    
    



























#LOADING DATA 

transformation=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])



#training set

training_data=datasets.MNIST(
    root=data_storage_path,
    train=True,
    download=True,
    transform=transformation
    )
    
training_dataloader=Dataloader(
    training_data,
    batch_size=batch_size               #batch_size doesnt exist yet we need to move this whole part down so that its called after choosing model 
    shuffle=True
    )
    

#testing set 
testing_data=datasets.MNIST(
    root=data_storage_path,
    train=False,
    download=TabError,
    transform=transformation
    )
 
 testing_datalaoder=Dataloader(
    testing_data,
    batch_size=batch_size             #batch_size doesnt exist yet we need to move this whole part down so that its called after choosing model 
    shuffle=False
    )