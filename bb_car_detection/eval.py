from dataset import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
coco_small = {1:"Car"}
coco_colors = {1:1}
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

# converts list to dict
def convert_list_to_dict(bblist):
    return {'x1' : bblist[0], 'y1' : bblist[1], 'x2': bblist[2], 'y2' : bblist[3]}

# find iou 
def get_iou(bb1, bb2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

transform = transforms.ToTensor()
def car_detection(image, nms_thresh, score_thresh):
    image = image.to(DEVICE)
    
    #image = pre_process(image=image).to(DEVICE)
    image = image.unsqueeze(0)
    with torch.no_grad():
        out = model2.forward(image)
        #print(out)
        # Apply nms to only keep desired boxes over threshold
        nms_keep = torchvision.ops.nms(out[0]['boxes'], out[0]['scores'], nms_thresh)
        out[0]['boxes'] = out[0]['boxes'][nms_keep]
        out[0]['scores'] = out[0]['scores'][nms_keep]
        out[0]['labels'] = out[0]['labels'][nms_keep]
        # Further remove via bbox score threshold
        score_keep = []
        for i, score in enumerate(out[0]['scores']):
            if score > score_thresh:
                score_keep.append(i)
        out[0]['boxes'] = out[0]['boxes'][score_keep]
        out[0]['scores'] = out[0]['scores'][score_keep]
        out[0]['labels'] = out[0]['labels'][score_keep]
        class_text = []
        # Convert numerical outputs to text
        for i in out[0]['labels'].cpu().numpy():
            if coco_small.get(i) is not None:
                class_text.append(coco_small.get(i))
            else:
                class_text.append("Other")
        out_labels = out[0]['labels'].detach().cpu().numpy()
        out_label_text = np.array(class_text)
        out_scores = out[0]['scores'].detach().cpu().numpy()
        out_bboxes = out[0]['boxes'].detach().cpu().numpy()
    return out_bboxes, out_labels, out_label_text, out_scores

# perform transforms
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

# Validation dataste
val_Dataset = Dataset(
    root_path_to_data=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\val',
    label_file=r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\data\bdd100k_labels_images_val.json',
    #label_file=r'C:\Users\anshul\Documents\school\ECE228\bb_car_detection\short_train.json',
    transform=get_transform(False),
    width=512,
    height=256
    )

model2= torch.load(r'./model_weights_mobilenet_backbone.pkl').to(DEVICE)
#model2= torch.load(r'model_weights_restnet50_backbone.pkl').to(DEVICE)
model2.eval()
iou_list = []

# Find avg iou on validation dataset
for i in range(0, len(val_Dataset)):
    ground_truth_boxes = (val_Dataset[i][1].get('boxes')).detach().cpu().numpy() #list of lists
    image_tensor = val_Dataset[i][0]
    pred_boxes, labels, classes, scores  = car_detection(image_tensor, 0.3, 0.7)

    for pred_box in pred_boxes:
        pred_dict = (convert_list_to_dict(pred_box))
        max_iou = 0
        for ground_truth_box in ground_truth_boxes:
            ground_truth_box_dict = (convert_list_to_dict(ground_truth_box))
            iou = (get_iou(pred_dict, ground_truth_box_dict))
            if(iou > max_iou):
                max_iou = iou
        iou_list.append(max_iou)

print(iou_list)
print(sum(iou_list) / len(iou_list))
