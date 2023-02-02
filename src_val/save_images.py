#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:07:22 2022

@author: eric
"""


import torch
#import matplotlib.pyplot as plt
from utils import unnorm, remove_axes
from train_segmentation import LitUnsupervisedSegmenter
# import png
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

print("Calentando motores...")
model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/cityscapes_vit_base_1.ckpt").cuda()
print("Model loaded!!")
# model=torch.load("../saved_models/cityscapes_vit_base_1.ckpt", map_location='cpu')

n_files= 8  
n_batch=64
for j in tqdm(range(n_files)):
    d=torch.load('stego_images/stego_images_'+str((j+1)*n_batch)+'.pt')

    #unpack dicts
    images=d["image"]
    cluster=d["cluster"]
    linear=d["linear"]

    for i in range(len(images)):    
        img=images[i]
        cluster_pred = cluster[i]
        linear_pred = linear[i]

        # unnormalize pixel values before printing
        save_image(unnorm(img),"images/image"+str(i+(j*n_batch))+".png")
        # PIL do not support RGB images in 0 to 1 range. 
        #label_cmap asocia el cluster label a un codigo de color
        #im = Image.fromarray((model.label_cmap[cluster_pred]).astype(np.uint8))
        
        #a = model.test_cluster_metrics.map_clusters(cluster_pred)
        im = Image.fromarray((model.label_cmap[cluster_pred]).astype(np.uint8))
        im.save("cluster/cluster"+str(i+(j*n_batch))+".png")

        im = Image.fromarray((model.label_cmap[linear_pred]).astype(np.uint8))
        im.save("linear/linear"+str(i+(j*n_batch))+".png")

print("Program finished...")









