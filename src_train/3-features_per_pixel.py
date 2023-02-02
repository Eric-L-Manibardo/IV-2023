#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:14:25 2022

@author: eric
"""
import torch
import numpy as np
import pickle
from tqdm import tqdm




n_items = 2975
images_per_file = 64
# n_files = int(np.ceil(n_items/images_per_file))
# =============================================================================
# recolectar features de los pixeles 
# =============================================================================
c=16

with open('pixel_pos/pixel_pos_per_cluster_'+str(c)+'.pkl', 'rb') as f:
    coor = pickle.load(f)
'''
dictionary with 2 keys: x and y, for 19998 images
foreach image, the coordinates of pixels of cluster 16 are stored
'''    
    
f_cluster =  list()
a=1
n_files=47
for n in tqdm(range(n_files)):
    
    features = torch.load('stego_features/stego_features_'+str((n+1)*images_per_file)+'.pt')['features']       
    #features = torch.load('/media/eric/Samsung_T5/stego_features/stego_features_train_extra_'+str((n+1)*images_per_file)+'.pt')['features']
    print('File read!')
    for i in range(len(features)):
        for j in tqdm(range(len(coor['x'][i+(images_per_file*n)]))):
            #features de cada uno de los j pixeles del cluster c de la imagen i
            f_cluster.append(features[i][:,coor['x'][i+(images_per_file*n)][j],coor['y'][i+(images_per_file*n)][j]].numpy())
            # imagen_ID.append(n+n*64)
    
    with open('features_per_pixel_cluster_'+str(c)+'/features_cluster_'+str(c)+'_part_'+str(n)+'.pkl', 'wb') as f:
        pickle.dump(np.array(f_cluster), f)   
    
    f_cluster =  list()
            
