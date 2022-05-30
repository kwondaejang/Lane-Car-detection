import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import faster_rcnn
import numpy as np

# # Smaller coco dataset to represent road objects
coco_small = {0:"Background", 1:"Car"}
coco_colors = {0:0, 1:1}
#model=faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
model = torch.load("2_trained.pkl")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# # num_classes = 2  # vehicle and background. 
# # in_features = model.roi_heads.box_predictor.cls_score.in_features
# # model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)
# print("Model Summary:\n", model)
model.eval()
num_classes = len(coco_small) + 1
# # box_colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k', 'w']
box_colors = np.random.uniform(0, 255, (num_classes, 3))
transform = transforms.ToTensor()


def predict(image, nms_thresh, score_thresh):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        out = model.forward(image)
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

""" 
Draw the bounding boxes on the image 

@parameter boxes: the bounding box coordinates
@parameter labels: the numerical object labels
@parameter labels_text: the textural object labels (classes)
@paramaeter image_path: the path to the image
@parameter save_path: the image to analyze
@return the new image with bounding boxes
"""
def drawBox(boxes, labels, labels_text, image):
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label_adjust = []
    # Convert labels to smaller coco labels defined above
    for label in labels:
        new_label = (coco_colors.get(label) if coco_colors.get(label) is not None 
                     else len(coco_small))
        label_adjust.append(new_label)
    for i, box in enumerate(boxes):
        color = box_colors[label_adjust[i]]
        cv2.rectangle(image, (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, labels_text[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, 
                    lineType = cv2.LINE_AA)
    return image
image = cv2.imread(r"C:\Users\anshul\Documents\school\ECE228\final_project\data\test_set\clips\0601\1494452441570881066\3.jpg")
#image = cv2.imread(r"C:\Users\anshul\Documents\school\ECE228\bb_car_detection\data\train\0000f77c-6257be58.jpg")
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
img_res = cv2.resize(img_rgb, (512, 256), cv2.INTER_AREA)
# diving by 255
#image_model = img_res.copy()
image_model = img_res/ 255.0
boxes, labels, classes, scores  = predict(image_model, 0.3, 0.7)
print(boxes, labels, classes, scores )
image_final = drawBox(boxes, labels, classes, img_res)
print(image_final.shape)
cv2.imwrite("test.png", image_final)



# transforms_ = A.Compose([
#     A.Normalize(),
#     ToTensorV2()
# ])


# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
# img_res = cv2.resize(img_rgb, (512, 256), cv2.INTER_AREA)
# img_test = img_res.copy()
# coords = [[ 18.0000,  90.3111, 142.8000, 173.1555], [202.8000,  78.5778, 363.2000, 157.1555]]
# def draw_bb(image, coords):
#     for i in coords:
#         pt1 = (int(i[0]), int(i[1]))
#         pt2 = (int(i[2]), int(i[3]))
#         cv2.rectangle(img_test, pt1, pt2, (255,0,0),2 )
#     return img_test



# image = cv2.imread(r"C:\Users\anshul\Documents\school\ECE228\bb_car_detection\data\train\0000f77c-6257be58.jpg")
# tensor = transforms_(image=image)
# input = (tensor["image"].unsqueeze(dim=0))
# input = input.to(DEVICE)

# model = torch.load('0_trained.pkl')
# model = model.eval()
# outputs = model(input)
# print(outputs)
# outputs = (outputs[0].get('boxes').detach().cpu().numpy())
# #print(outputs)
# cv2.imwrite("test.png",draw_bb(img_test, coords=coords))