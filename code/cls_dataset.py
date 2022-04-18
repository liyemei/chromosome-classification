#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 23:50:15 2022

@author: root
"""

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import cv2
import matplotlib

def CLS_Loader(path):

    img=cv2.imread(path,0)
    img=cv2.resize(img,(128,188),interpolation=cv2.INTER_CUBIC)/255

    img= Image.fromarray(img)
    return img
    
class CLS_Dataset (Dataset):

    def __init__(self, txt, transform=None, loader=CLS_Loader,mode='Train'):
        self.mode=mode
        with open(txt, 'r') as fh:
            samples = []
            labels=[]
            for line in fh:
                line = line.strip('\n')  
                line = line.rstrip() 
                words = line.split( )  
                samples.append((words[0], int(words[1]))) 
                labels.append(int(words[1]))
 
        [samples_train_index, samples_test_index]= list(StratifiedShuffleSplit(
                                                                n_splits=1,test_size=0.2,random_state=123
                                                                ).split(samples,labels))[0]
        
        samples_trains = [samples[i] for i in samples_train_index]
        samples_test = [samples[i] for i in samples_test_index]
       
        samples_trains_targets = [s[1] for s in samples_trains]
        [samples_train_index, samples_val_index]= list(StratifiedShuffleSplit(
                                                            n_splits=1, test_size= 0.25,random_state=123
                                                             ).split(samples_trains, samples_trains_targets))[0]
       
        samples_train = [samples_trains[i] for i in samples_train_index]
        samples_val = [samples_trains[i] for i in samples_val_index]
#      
        if self.mode=='Train':
            self.imgs= samples_train
        if self.mode=='Test':
            self.imgs=samples_test 
        if self.mode=='Val':
            self.imgs=samples_val
        self.transform = transform
        
        self.loader = loader

    def __getitem__(self, index):
        img, label = self.imgs[index]
 
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

