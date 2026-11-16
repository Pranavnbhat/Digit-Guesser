import torch

from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import v2

from pathlib import Path






#SETTINGS/ BASIC SETUP

if torch.cuda.is_available():   device="cuda"
else:   device="cpu"
print('Using ', device, ' to run the model')

base_dir=Path(__file__).resolve().parent
model_path=base_dir / "model.pth"
data_storage_path=base_dir / "data"
img_path=base_dir/"images"




#MODEL
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.flatten=nn.Flatten()
        self.stack=nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self,images):
        images=self.flatten(images)
        raw_scores=self.stack(images)
        return raw_scores




#TRAINING FUCNTION 



def train(model,dataloader,optimizer):
    
    model.train()    #swithces model to training mode only 
    loss_calc=nn.CrossEntropyLoss()
    
    data_size=len(dataloader.dataset)                          #data_size = len(training_data) doesnt work as training_data isnt defined
    
    for current_batch, (images,labels) in enumerate(dataloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        #FORWARD PASS 
        prediction=model(images)
        loss=loss_calc(prediction,labels)
        
        
        #BACKPASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        #report progress
        if current_batch%batch_size==0: 
            
            print('Batch done')
            print('Loss is ', loss.item())
            print('Images processed =', min((current_batch+ 1) * len(images), data_size))






   
#TESTING FUCNTON

 
def test(model,dataloader):
    model.eval()   #just sets model to eval mode similiar to model.test()
    
    number_of_batches=len(dataloader)
    data_size=len(dataloader.dataset)
    
    correct=0
    
    for images, labels in dataloader:
        
        images=images.to(device)
        labels=labels.to(device)
        
        prediction=model(images)
        
        #to find accuracy we first convert raw score to values then compare with labels
        prediction=prediction.argmax(1)
        corrected_predictions= (prediction==labels)
        corrected_predictions = corrected_predictions.type(torch.float)
        
        correct += corrected_predictions.sum().item()
        
    accuracy=correct/data_size
    
    return accuracy
        
    
    





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
    






#TRAINING MODE   
def train_model(epochs,lr,batch_size):
    
    print('Loading data to train model on.............')
    training_dataloader, testing_dataloader=load_data(batch_size)
    
    
    print('Data loaded successfully.\nTraining a new model...............')
    
    model = Model().to(device)
    
    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    
    current_epoch=0
    
    for epoch in range(epochs):
        train(model,training_dataloader,optimizer)
        
        current_epoch+=epoch
        print("Epoch", current_epoch+1, "/", epochs)
        
        accuracy=test(model, training_dataloader)
        
        
    print("\nTraining complete")
    
    #GET OLDER MODEL ACCURACY 
    #COMPARE ACCURACIES 
    #SAVE MODEL 
    previous_accuracy=0.00001   #just for debugging rn 
    if previous_accuracy is not None:
        
        if accuracy>previous_accuracy:
            torch.save(model.state_dict(), model_path)
            print("Newer model's accuracy is higher by x%, saving newer model to ", model_path)
        else: print("Older model's accuracy is higher by x%, not saving the newer model")
    
    else:   torch.save(model.state_dict(), model_path); print("Model saved to ", model_path)





#TESTING MODE





  
#MAIN MENU AND DEFAULTS 
print("\n==============================")
print("       DIGIT GUESSER")
print("==============================")

print("1. Train model")
print("2. Test model using images")
print("3. Change device")

choice = input("\nChoose an option (1/2/3): ").strip()

if choice=="1":
    #CHECK PREVIOUS SAVED DATA 
    #ASK(IF NO SAVES) FOR NEW VALUES 
    
    
    
    epoch=int(input('Enter number of epochs '))
    lr=float(input('Enter learning rate '))
    batch_size=int(input('Enter batch size ' ))
    
    train_model(epoch,lr,batch_size)
    
elif choice=="2":
    #ASK IMAGE LOCATION 
    test_image() 
    


elif choice=="3":
    if device=="cuda":
        device="cpu"
    elif torch.cuda.is_available(): device="cuda"
    else: print('Switching device unavailable')
    
    print(f"Using {device} device")
    
    
    

else: print('Invalid option')

