import json
from logging import root
import torch
import os 
import numpy as np
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self,root_path_to_data, label_file, width, height, transform=None):
        self.root = root_path_to_data
        self.label_file_data = json.load(open(label_file)) 
        self.transform = transform
        self.width = width
        self.height = height

        self.image_data = {} # dicts of dicts
        counter =0 
        for image_dict in self.label_file_data:
            if((image_dict['attributes'].get('weather') == 'clear') and (image_dict['attributes'].get('scene') == "highway") and (image_dict['attributes'].get('timeofday') == "daytime") ):
                box2d_list = []
            
                for i in ((image_dict['labels'])):
                    if(i['category'] == 'car'):
                        box2d_list.append(i['box2d'])
                   
                if(len(box2d_list) != 0):
                    self.image_data[counter] = {
                        "image_name" : image_dict['name'],
                        "carbb_coords": box2d_list 
                    }
                    counter +=1


    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_data[idx].get('image_name'))
        boxes = []
    

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        wt = img.shape[1]
        ht = img.shape[0]

        # Rescale bounding boxes after resize
        bbox_labels_list = self.image_data[idx].get('carbb_coords')
        num_objs = len(bbox_labels_list)
        for i in bbox_labels_list:
            x1 = int(i.get('x1'))
            y1 = int(i.get('y1'))
            x2 = int(i.get('x2'))
            y2 = int(i.get('y2'))

            x1_scaled = (x1/wt)*self.width
            x2_scaled = (x2/wt)*self.width
            y1_scaled = (y1/ht)*self.height
            y2_scaled = (y2/ht)*self.height

            boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

        boxes= torch.as_tensor(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # only one class
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # to be used if zero

        targets = {}
        targets['boxes'] =  boxes
        targets["labels"] = labels
        targets["image_id"] = image_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd

        if self.transform:
            sample = self.transform(image = img_res,
                                bboxes = targets['boxes'],
                                labels = labels)
            img_res = sample['image']
            targets['boxes'] = torch.Tensor(sample['bboxes'])

        
        return img_res, targets



