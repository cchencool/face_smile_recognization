#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

img_folder = './data/'

compressed_img_size = (45, 25)
def load_images(folder):
    (images, lables) = ([], [])
    lable = 0
    for (subdirs, dirs, files) in os.walk(folder):
        print(subdirs, dirs)
        for subdir in dirs:
            subjectpath = os.path.join(folder, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                img = cv2.imread(path, 0) 
                img = cv2.resize(img,compressed_img_size)
                images.append(img)
                lables.append(int(lable))
            lable += 1
        
        return images, lables

X, Y = load_images("./data/")

# (X,Y) = [np.array(lis) for lis in [X, Y]]
# Y = pd.get_dummies(Y) #converting labels to one-hot, Used Pandas for it.
