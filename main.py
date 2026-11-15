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



def train(model,dataloader,optimizer):
    
    model.train()    #swithces model to training mode only 
    loss_calc=nn.CrossEntropyLoss()
    
    data_size=len(training_data)
    
    for current_batch, (images,labels) in enumerate(dataloader):
        
        images.to(device)
        labels.to(device)
        
        #FORWARD PASS 
        prediction=model(images)
        loss=loss_calc(prediction,y)
        
        
        #BACKPASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        #report progress
        if current_batch%batch_size==0: 
            
            print('Batch done')
            print('Loss is ', loss.item())
            print('Images processed =', min((current_batch+ 1) * len(X), data_size))






   
#TESTING FUCNTON

 
def test(model,dataloader):
    model.eval()   #just sets model to eval mode similiar to model.test()
    
    number_of_batches=len(dataloader)
    data_size=len(training_data
    
    for images, labels in dataloader:
        
        images.to(device)
        labels.to(device)
        
        prediction=model(images)
        
        #to find accuracy we first convert raw score to values then compare with labels
        prediction=prediction.argmax(1)
        corrected_prediction= (prediction==labels)
        correct_predictions = correct_predictions.type(torch.float)
        
        correct += correct_predictions.sum().item()
        
    accuracy=correct/data_size
    
    retrun accuracy
        
    
    





#LOADING DATA 


def load_data(batch_size):

    transformation = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # training set
    training_data = datasets.MNIST(
        root=data_storage_path,
        train=True,
        download=True,
        transform=transformation
    )

    training_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True
    )

    # testing set
    testing_data = datasets.MNIST(
        root=data_storage_path,
        train=False,
        download=True,
        transform=transformation
    )

    testing_dataloader = DataLoader(
        testing_data,
        batch_size=batch_size,
        shuffle=False
    )

    return training_dataloader, testing_dataloader
    









    
#MAIN MENU AND DEFAULTS 
print("\n==============================")
print("       DIGIT GUESSER")
print("==============================")

print("1. Train model")
print("2. Test model using images")
print("3. Change device")

choice = input("\nChoose an option (1/2/3): ").strip()


if choice=="3":
    if device=="cuda":
        device=="cpu"
    elif torch.cuda.is_available(): device=="cuda"
    else: print('Switching device unavailable')
    
    print(f"Using {device} device")
    
    
    
if choice=="1":
    #CHECK PREVIOUS SAVED DATA 
    #ASK(IF NO SAVES) FOR NEW VALUES 
    #RUN MODEL 
    #SAVE MODEL 
    
if choice=="2":
    #ASK IMAGE LOCATION 
    #RUN TESTING MODEL 
    







#TRAINING MODE 




#TESTING MODE






