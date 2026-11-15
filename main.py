import torch

from torch import nn 
from torch.utis.data import Dataloader
from torchvision import datasets 
from torchvision.transforms import V2

from pathlib import Path






#SETTINGS/ BASIC SETUP

if torch.cuda_is_available():   device="cuda"
else:   device="cpu"
print('Using ', device' to run the model')

base_dir=Path(__file__).reolve().parent
model_path=base_dir/"model.pth"




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
        
    def forward(self,x):
        x=self.flatten(x)
        raw_score=self.stack(x)
        return raw_score
            