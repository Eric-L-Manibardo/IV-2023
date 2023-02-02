#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:19:14 2022

@author: eric
"""
from train_segmentation import LitUnsupervisedSegmenter
from torchvision.transforms.functional import to_tensor
from utils import get_transform,unnorm
from tqdm import tqdm
import torch.nn.functional as F
from crf import dense_crf
import torch
from torchvision import datasets, transforms
from multiprocessing import Pool
from torchvision.utils import save_image


resolution = 320
# normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
'predict': get_transform(resolution,False, "center")
}
batch_size = 8
dataset = {'predict' : datasets.ImageFolder('../datadrive/pytorch-data/val_GT', data_transforms['predict'])}
test_loader = {'predict': torch.utils.data.DataLoader(
    dataset['predict'],
    batch_size = batch_size,
    shuffle=False,
    num_workers=12,
    pin_memory=True)}
print("Inference dataset loaded!!")

cont =0
for img, label in tqdm(test_loader["predict"]):
         
    for i in range(len(img)):
        save_image(unnorm(img[i]),"GT/gt_"+str(cont)+".png")
        cont = cont + 1


       
       
       
       
       
       
       
