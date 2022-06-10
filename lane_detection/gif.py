from telnetlib import X3PAD
import imageio
import pathlib
import os
import torchvision.transforms as transforms
import cv2 
import torch
import numpy as np
import time
import albumentations as A
from model.model import *
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
mapping_value_to_text = {1:"Car"}
color_mapping = {1:1}
IMAGE_HEIGHT = 256
IMAGE_WIDTH= 512
LANE_DEP_THRES = 300
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
transform = transforms.ToTensor()


# perform car detection
def car_detection(image, nms_thresh, score_thresh):
    image = transform(image).type(torch.float32).to(DEVICE)
    
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
            if mapping_value_to_text.get(i) is not None:
                class_text.append(mapping_value_to_text.get(i))
            else:
                class_text.append("Other")
        out_labels = out[0]['labels'].detach().cpu().numpy()
        out_label_text = np.array(class_text)
        out_scores = out[0]['scores'].detach().cpu().numpy()
        out_bboxes = out[0]['boxes'].detach().cpu().numpy()
    return out_bboxes, out_labels, out_label_text, out_scores

# draw bounding boxes on the frame
def drawBox(boxes, labels, image):
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label_adjust = []
    # Convert labels to smaller coco labels defined above
    for label in labels:
        new_label = (color_mapping.get(label) if color_mapping.get(label) is not None 
                     else len(mapping_value_to_text))
        label_adjust.append(new_label)
    for i, box in enumerate(boxes):
        color = (255,20,147)
        cv2.rectangle(image, (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), color, 1)

    return image

# preprocess transfroms
pre_process = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(),
    ToTensorV2(),
    
])

# driver alert algorthim
def lane_depature_warning(pred_frame):
    # produce mask from region of interest
    roi = np.array([[[128,256],[384,256],[256,110]]], dtype=np.int32)
    mask = np.zeros(pred_frame.shape, dtype=np.uint8)
    
    # calculate the number of lane pixels in the roi
    cv2.fillPoly(mask, roi, (1))
    masked_image = cv2.bitwise_and(pred_frame, mask)
    return np.sum(masked_image)
    

# perfrom lane detection inference on a given frame
def lane_detection(model, frame):
    lane_dep = False
    tensor = pre_process(image=frame)
    input = (tensor["image"].unsqueeze(dim=0))
    input = input.to(DEVICE)

    preds = (model(input))
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds_numpy = preds.cpu().detach().numpy().astype(np.uint8)
    preds_numpy = np.squeeze(preds_numpy)

    num_of_pixels = lane_depature_warning(preds_numpy)

    if(num_of_pixels > LANE_DEP_THRES):
        lane_dep = True
    frame[preds_numpy==1] = (36,255,12)

    return preds_numpy, frame, lane_dep


# Define models
#model = UNET(out_channels=1, features=[32,64,128,256]).to(device=DEVICE)
model = SegNet().to(device=DEVICE)
model.load_state_dict(torch.load(r'C:\Users\anshul\Documents\school\ECE228\final_project\model_weights\best_segnet_ftl_iou\epoch_34_loss_0.6171251664079469.tar')['state_dict'])
model = model.eval()

model2= torch.load(r'C:\Users\anshul\Documents\school\ECE228\Lane-Car-detection\bb_car_detection\model_weights_mobilenet_backbone.pkl').to(DEVICE)
model2.eval()

# Define path to images
top_level_folder_paths = [r'C:\Users\anshul\Documents\school\ECE228\final_project\data\test_set\clips\0531\1492635549292553036',r'C:\Users\anshul\Documents\school\ECE228\final_project\data\test_set\clips\0531\1492635073510558516']
images = []
car_fps = []
lane_fps = []
for paths in top_level_folder_paths:
    # Get list of all files in a given directory sorted by name
    flist = [p for p in pathlib.Path(paths).iterdir() if p.is_file()]
    sorted_list = sorted(flist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))


    # For each file peform car and lane inference and log the time it takes to do each
    # this is used to calc FPS
    for filename in sorted_list:
        frame = imageio.imread(filename)
        frame = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_NEAREST )
        frame2 = frame/255
        t0 = time.time()
        preds, overlay_image, is_lane_dep = lane_detection(model, frame)
        t1 = time.time()
        boxes, labels, classes, scores  = car_detection(frame2, 0.3, 0.7)
        t2 = time.time()
        # if lane depature overlay "lane depature" text on final output image
        if(is_lane_dep):
            overlay_image =cv2.putText(img=np.copy(overlay_image), text="lane departure", org=(0,25), fontFace=3, fontScale=1, color=(255,0,0), thickness=2)
        image_final = drawBox(boxes, labels, overlay_image)
        images.append(image_final)
        car_fps.append(t2-t1)
        lane_fps.append(t1-t0)

car = car_fps[1:]
lane = lane_fps[1:]
# calc FPS
car_avg_fps = 1 / (sum(car) / len(car))
lane_avg_fps = 1 / (sum(lane) / len(lane))
print(car_avg_fps)
print(lane_avg_fps)
imageio.mimsave('./test.gif', images)


