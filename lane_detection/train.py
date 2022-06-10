import numpy as np
import torch
from dataset.dataset import tusimpleDatasetReg
from model.model import *
from utils.utils import *
from model.loss import *
import cv2

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
LEARNING_RATE = .0001
BATCHSIZE=8
NUM_EPOCHS = 30
torch.manual_seed(1)


# Does validation on val dataset
# This is done after each traning epoch
# Metrics measured loss
def model_validate(model, loader, loss_fn1, loss_fn2):
    # Put model in evaluate mode
    print("VALIDATION")
    model.eval()
    val_loss = 0
    loop = tqdm(loader)
    # We dont need gradiant calculations in validation
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loop):

            # Get inputs and gt label
            inputs = inputs.to(device=DEVICE)
            labels = labels.unsqueeze(dim=1).float().to(device=DEVICE)

            # Run inputs through model
            output= model(inputs)
            loss_1 = loss_fn1(output, labels) # ftl loss
            loss_2 = loss_fn2(output, labels) # iou loss
            total_loss= loss_1 + loss_2

            # keep track of loss
            val_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

    
    val_loss /= (len(loader.dataset))

    print('\nAverage loss: {:.4f}'.format(val_loss))
    
    return val_loss

# does one epoch of traning
def model_train(model, loader ,optimizer,loss_fn1, loss_fn2):
    model.train()
    loop = tqdm(loader)

    for batch_idx, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(device=DEVICE)
        labels = labels.unsqueeze(dim=1).float().to(device=DEVICE)
        optimizer.zero_grad()
        preds = model(inputs)
        loss_1= loss_fn1(preds, labels)
        loss_2 = loss_fn2(preds,labels)
        total_loss = loss_1 + loss_2
        total_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=total_loss.item(), loss_ftl=loss_1.item(), loss_iou=loss_2.item())

# transforms on train dataset
train_transforms_ = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(),
    ToTensorV2(), 
])

# transforms on validation dataset
val_transforms = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(),
    ToTensorV2(), 
])

# Specify dataset and loaders
root = 'C:/Users/anshul/Documents/school/ECE228/final_project/dataset/'
train_set = tusimpleDatasetReg(root_path_to_data=root, dataclass='train', transforms=train_transforms_)
valid_set = tusimpleDatasetReg(root_path_to_data=root, dataclass='valid', transforms=val_transforms)

data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
data_loader_valid = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=0)


model = SegNet().to(DEVICE)
#model = UNET(out_channels=1, features=[32,64,128,256]).to(device=DEVICE)

# define optimizer, schduler, and loss functinos
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = FocalTverskyLoss().to(DEVICE)
criterion_iou = IoULoss()


best_val_loss = np.inf

# Start Training
print("Begin Training")
for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model_train(model,data_loader_train,optimizer,criterion,criterion_iou )

    # validate
    val_loss = model_validate(model, data_loader_valid, criterion, criterion_iou)

    # save if new lowest validation loss
    if(val_loss < best_val_loss):
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }

        save_checkpoint(checkpoint, filename='./saved_models/epoch_{}_loss_{}.tar'.format(epoch, val_loss))
        best_val_loss = val_loss
    
    # used to output images as training progress
    #save_predictions_as_imgs(data_loader_valid, model=model, epoch=epoch)

    #step the scheduler
    scheduler.step()