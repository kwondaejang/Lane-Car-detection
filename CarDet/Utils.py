import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from queue import Queue
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.data import Dataset
from IPython.display import display
import PIL

from modelWrapper import *
from Box_det import *
from Training_Case import *

training_dataset_path = './kaggle/input/car-object-detection/data/training_images'
testing_dataset_path = './kaggle/input/car-object-detection/data/testing_images'
box_df = pd.read_csv('./kaggle/input/car-object-detection/data/train_solution_bounding_boxes (1).csv')

def df2training_list(df):
    trn_dict = dict()

    for i in box_df.itertuples():
        if not i.image in trn_dict:
            trn_case = TrainingCase(training_dataset_path + '/' + i.image)
            trn_dict[i.image] = trn_case
        else:
            trn_case = trn_dict[i.image]
        trn_case.add_box(i)

    trn_list = [val for (key,val) in trn_dict.items()]

    return trn_list

def build_model():
    model = MobileNetV2(weights='imagenet',include_top=False)
    x = model.outputs[0]
    print('xshape: ', x.shape)
    x = Conv2D(2,1)(x)
    
    model = Model(model.inputs,x)
    model.compile('adam',loss = 'mse')
    return model

def draw_outputs(outputs):
    for out in outputs:
        out = out * 1 # This operation should case boolean matrix to numeric.
        min_val = out.min()
        max_val = out.max()
        out = (out - min_val) / (max_val - min_val)
        img = PIL.Image.fromarray((out*255).astype(np.uint8),'L')
        w,h = img.size
        img = img.resize((w*4,h*4))
        display(img)

def load_test_images(path):
    cases = []
    for filename in os.listdir(path):
        file_path = path + '/' + filename
        case = TrainingCase(file_path)
        cases.append(case)
    
    return cases

def get_box(model,case):
    prediction = model.predict(case)
    mask = prediction[2].copy()
    h,w = mask.shape
    box = []
    
    def walk_on_box(sx,sy):
        box = Box_Det(sx,sy,sx,sy)
        q = Queue()
        q.put((sx,sy))
        while not q.empty():
            x,y = q.get()
            box.box_resizing(x+1,y+1)
            mask[y,x] = False
            if y+1 < h and mask[y+1,x]:
                q.put((x,y+1))
            if y-1 >= 0 and mask[y-1,x]:
                q.put((x,y-1))
            if x+1 < w and mask[y,x+1]:
                q.put((x+1,y))
            if x-1 >= 0 and mask[y,x-1]:
                q.put((x-1,y))
        
        return box.box_scaling(32)
    
    for y in range(h):
        for x in range(w):
            if mask[y,x]:
                box.append(walk_on_box(x,y))
    
    case.boxes = box
