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
import torch
#import matplotlib.pyplot as plt
from utils import unnorm, remove_axes

# import png
from PIL import Image
import numpy as np
from torchvision.utils import save_image


print("Calentando motores...")
model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/cityscapes_vit_base_1.ckpt").cuda()
print("Model loaded!!")

resolution = 320
# normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
'predict': get_transform(resolution,False, "center")
}
batch_size = 8
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

        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()
         
        for i in range(len(img)):
            single_img = img[i].cpu()
            a.append(single_img)
            b.append(dense_crf(single_img, linear_probs[i]).argmax(0))#numpy
            c.append(dense_crf(single_img, cluster_probs[i]).argmax(0))#numpyS
            features.append(code[i].cpu())#pass to cpu for freeing gpu space
        #contamos los batch pasados
        count=count+1
        if count%8==0:
                print('\n Image # '+str(count*batch_size))
                torch.save({"features":features},'stego_features/stego_features_p'+str(count*batch_size)+'.pt') 
                torch.save({"image": a, "linear":b,"cluster":c},'stego_images/stego_images_p'+str(count*batch_size)+'.pt')     
                features = list()  
                a,c = list(), list()

            
tb_metrics = {
    **model.test_linear_metrics.compute(),
    **model.test_cluster_metrics.compute(),
    }
       #torch.cuda.empty_cache()

#torch.save({"image": a, "linear":b,"cluster":c},"results_stego_images_test_original.pt")

# guardamos un ultima vez
torch.save({"features":features},'stego_features/stego_features_p'+str(count*batch_size)+'.pt') 
torch.save({"image": a, "linear":b,"cluster":c},'stego_images/stego_images_p'+str(count*batch_size)+'.pt')

n_files= 1 #47  
n_batch=64
for j in tqdm(range(n_files)):
    d=torch.load('stego_images/stego_images_p'+str((j+1)*n_batch)+'.pt')

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

        asociation = model.test_cluster_metrics.map_clusters(cluster_pred)
        im = Image.fromarray((model.label_cmap[asociation]).astype(np.uint8))
        im.save("cluster/cluster"+str(i+(j*n_batch))+".png")


        im = Image.fromarray((model.label_cmap[linear_pred]).astype(np.uint8))
        im.save("linear/linear"+str(i+(j*n_batch))+".png")

print("Program finished...")  
       
       
       
       
       
       
       
       
       
       
       
