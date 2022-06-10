from operator import mod
import torch
from dataset.dataset import tusimpleDatasetReg
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from model.model import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# function that loops through dataset and runs predication and calculates
# f1 score and accuracy between prediction and GT
def model_eval(model, data_loader):
    loop = tqdm(data_loader)
    f1_score_val = 0
    accuracy_score_val = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loop):

            # Get inputs and gt label
            inputs = inputs.to(device=DEVICE)
            labels = labels.unsqueeze(dim=1).float().to(device=DEVICE)
            target = labels.detach().cpu().numpy().ravel()

            predection = model(inputs)
            predection = torch.sigmoid(predection).detach().cpu().numpy().ravel()

            f1_score_val += f1_score((target > 0.5).astype(np.int64), (predection > 0.5).astype(np.int64), zero_division=1)
            accuracy_score_val += accuracy_score((target > 0.5).astype(np.int64), (predection > 0.5).astype(np.int64))

    f1_score_val /= (len(data_loader.dataset))
    accuracy_score_val /= (len(data_loader.dataset))


    print('\nAverage f1 score: {:.4f}'.format(f1_score_val * 100))
    print('\nAverage accuracy score: {:.4f}'.format(accuracy_score_val * 100))

        

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

# Specifiy Transforms
val_transforms = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(),
    ToTensorV2(),
    
])

# dataset and dataloader
root = 'C:/Users/anshul/Documents/school/ECE228/final_project/dataset/'
test_set = tusimpleDatasetReg(root_path_to_data=root, dataclass='test', transforms=val_transforms)
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


#model = UNET(out_channels=1, features=[32,64,128,256]).to(device=DEVICE)
model = SegNet().to(device=DEVICE)
model.load_state_dict(torch.load(r'C:\Users\anshul\Documents\school\ECE228\final_project\best_segnet_FTL\epoch_24_loss_0.033042417630424786.tar')['state_dict'])
model = model.eval()
# get avg f1 score and accuracy
model_eval(model, data_loader_test)


