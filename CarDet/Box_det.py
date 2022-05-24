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

class Box_Det:
    def __init__(self,x_min,y_min,x_max,y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    
    def box_tuple(tup):
        return Box_Det(tup.xmin,tup.ymin,tup.xmax,tup.ymax)
    
    def box_scaling(self,scale_mul):
        return Box_Det(
            self.x_min * scale_mul,
            self.y_min * scale_mul,
            self.x_max * scale_mul,
            self.y_max * scale_mul
        )
    
    def box_resizing(self,x,y):
        if self.x_min > x:
            self.x_min = x
        if self.x_max < x:
            self.x_max = x
        if self.y_min > y:
            self.y_min = y
        if self.y_max < y:
            self.y_max = y