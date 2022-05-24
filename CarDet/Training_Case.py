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
from Box_det import *

def img2Tensor(img_arr):
    x,y,z = img_arr.shape
    return img_arr.reshape((1,x,y,z))

class TrainingCase:
    def __init__(self,img_path):
        self.img_path = img_path
        self.boxes = []
    
    def add_box(self,tup):
        self.boxes.append(Box_Det.box_tuple(tup))
    
    def get_image(self):
        img = load_img(self.img_path)
        img_arr = img_to_array(img)
        return img_arr
    
    def draw_image_with_boxes(self):
        img = load_img(self.img_path)
        img_arr = img_to_array(img)
        h,w = img_arr.shape[:2]
        
        def point(y,x,color):
            x = int(x)
            y = int(y)
            if x >= 0 and x < w and y >= 0 and y < h:
                img_arr[y,x,:] = color
        
        for box in self.boxes:
            if box.x_min-1 >= 0:
                for y in range(int(box.y_min),int(box.y_max)):
                    point(y, box.x_min-1,(0,255,0))
                    point(y, box.x_min  ,(0,255,0))
                    point(y, box.x_min+1,(0,255,0))
                    point(y, box.x_max-1,(0,255,0))
                    point(y, box.x_max  ,(0,255,0))
                    point(y, box.x_max+1,(0,255,0))
                for x in range(int(box.x_min),int(box.x_max)):
                    point(box.y_min-1, x,(0,255,0))
                    point(box.y_min  , x,(0,255,0))
                    point(box.y_min+1, x,(0,255,0))
                    point(box.y_max-1, x,(0,255,0))
                    point(box.y_max  , x,(0,255,0))
                    point(box.y_max+1, x,(0,255,0))
        
        img = PIL.Image.fromarray(img_arr.astype(np.uint8),'RGB')
        img.show()
        display(img)

    def get_img(self):
        img = load_img(self.img_path)
        img_w, img_h = img.size
        
        h,w = int(img_h/32),int(img_w/32)
        out_arr = np.concatenate((np.full((1,h,w,1),-1,dtype=np.float64),np.ones((1,h,w,1),dtype=np.float64)),axis=3)
        
        
        for box in self.boxes:
            x_min,y_min,x_max,y_max = int(box.x_min / 32),int(box.y_min / 32),int(box.x_max / 32),int(box.y_max / 32)
            for y in range(y_min,y_max):
                for x in range(x_min,x_max):
                    if y < 0 or x < 0 or y >= h or x >= w:
                        continue
                    
                    out_arr[0,y,x,0] = 1
                    out_arr[0,y,x,1] = -1
        
        return out_arr
        
    def get_answer_as_outputs(self):
        img = self.get_img()
        return img[0,:,:,0],img[0,:,:,1]

    def get_inputs(self):
        return img2Tensor(self.get_image())