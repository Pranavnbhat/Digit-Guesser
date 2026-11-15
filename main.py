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

