#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:19:14 2022

@author: eric
"""
from train_segmentation import LitUnsupervisedSegmenter
from torchvision.transforms.functional import to_tensor
from utils import get_transform
from tqdm import tqdm
import torch.nn.functional as F
from crf import dense_crf
import torch
from torchvision import datasets, transforms
from multiprocessing import Pool
import numpy as np

print("Calentando motores...")
model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/cityscapes_vit_base_1.ckpt").cuda()
print("Model loaded!!")

resolution = 320
# normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
'predict': get_transform(resolution,False, "center")
}
batch_size = 4
dataset = {'predict' : datasets.ImageFolder('../datadrive/pytorch-data/prueba', data_transforms['predict'])}
test_loader = {'predict': torch.utils.data.DataLoader(
    dataset['predict'],
    batch_size = batch_size,
    shuffle=False,
    num_workers=12,
    pin_memory=True)}
print("Inference dataset loaded!!")

a,b,c, features = list(), list(), list(), list()
count = 0
for img, label in tqdm(test_loader["predict"]):
    # no grad for inference mode so no previous computations are stored in GPU, hence freeing GPU space
    with torch.no_grad():

        code1 = model(img.cuda())
        code2 = model(img.flip(dims=[3]).cuda())
        code  = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu() # size of a linear_prob -> [27,320,320] [class,H,W]
        #cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()
        #print('SIZE of Linear probs of image #1:')
        #print(linear_probs[0].size())
        #print('Linear probs of image #1:')
        #print(linear_probs[0])
        
        for i in range(len(img)):
            single_img = img[i].cpu()
            linear_pred = dense_crf(single_img, linear_probs[i]).argmax(0) # size of a linear_pred -> [27,320,320] [class,H,W]
            #
            #print('SIZE of Linear pred of image #1:')
            #print(np.shape(linear_pred))
            print('linear predictions of image #1:')
            print(linear_pred[0,0])
            print('mapeo colores')
            print(model.label_cmap[linear_pred[0,0]])
            print(model.label_cmap[cluster_pred])
            break

        break



       
       
       
       
       
       
       
       
       
       
       
       
