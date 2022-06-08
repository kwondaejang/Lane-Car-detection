import torch
from tqdm import tqdm # progress bar
import math
import sys
import numpy as np
import pandas as pd
from detection.engine import evaluate, train_one_epoch
from detection.utils import collate_fn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import Dataset
from torchvision.models.detection import faster_rcnn


DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

# Tranform function
def get_transform(train):
  if train:
    return A.Compose(
      [
        A.HorizontalFlip(0.5),
        # ToTensorV2 converts image to pytorch tensor without div by 255
        ToTensorV2(p=1.0) 
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return A.Compose(
      [ToTensorV2(p=1.0)],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

# does 1 epoch of training
def train(model, optimizer, loader, device, epoch):
    model.train()
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        #print(targets)
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

# define train dataset
train_Dataset = Dataset(
    root_path_to_data=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\train',
    label_file=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\bdd100k_labels_images_train.json',
    #label_file=r'C:\Users\anshul\Documents\school\ECE228\bb_car_detection\short_train.json',
    transform=get_transform(True),
    width=512,
    height=256
    )
# Define validation dataset
valid_Dataset = Dataset(
    root_path_to_data=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\val',
    label_file=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\bdd100k_labels_images_val.json',
    #label_file=r'C:\Users\anshul\Documents\school\ECE228\bb_car_detection\short_valid.json',
    transform=get_transform(False),
    width=512,
    height=256
    )

# train and valid dataloaders
data_loader_train = torch.utils.data.DataLoader(
        train_Dataset, batch_size=4, shuffle=True,
        collate_fn=collate_fn)
data_loader_valid = torch.utils.data.DataLoader(
    valid_Dataset, batch_size=1, shuffle=False,
    collate_fn=collate_fn)


# Define model
num_classes = 2  # vehicle and background. 
model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
#model =faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_FPN_Weights)


in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
print(model)
model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# perform traning
num_epochs = 10
for epoch in range(num_epochs):
    #train(model, optimizer, data_loader_train, DEVICE, epoch)
    train_one_epoch(model,optimizer,data_loader_train,DEVICE,epoch, print_freq=10 )
    lr_scheduler.step()
    evaluate(model, data_loader_valid, device=DEVICE)
    torch.save(model, f"{epoch}_trained.pkl")


print("Done training")