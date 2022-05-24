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

class modelWrapper:
    def __init__(self,model):
        self._model = model
    
    @staticmethod
    def _normalize(matrix,min_val,max_val):
        return (matrix - min_val) / (max_val - min_val)
    
    def predict(self,case):
        inputs = case.get_inputs()
        results = self._model.predict(inputs)
        
        outs1 = results[0,:,:,0]
        outs2 = results[0,:,:,1]
        
        min_val = min(outs1.min(),outs2.min())
        max_val = min(outs1.max(),outs2.max())
        
        n_outs1 = modelWrapper._normalize(outs1,min_val,max_val)
        n_outs2 = modelWrapper._normalize(outs2,min_val,max_val)
        
        return outs1,outs2,(n_outs1 > n_outs2)
    
    def __call__(self,case):
        return self.predict(case)
