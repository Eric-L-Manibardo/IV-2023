#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:23:46 2022

@author: eric
"""
import numpy as np
import pickle
from tqdm import tqdm
import imageio
import torch
from PIL import Image

c=16

with open('kmeans_models/kmeans_'+str(c)+'.pkl', 'rb') as f:
# with open('kmeans_models/kmeans_16A.pkl', 'rb') as f:
    model = pickle.load(f)

with open('kmeans_models/scaler_'+str(c)+'.pkl', 'rb') as f:
# with open('kmeans_models/scaler_16A.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
'''
usar esta paleta para experimentation
''' 

# segunda paleta para guardar finalmente los colores
custom_palette =np.array([[255,128,0],[233,128,233]])


color_map = np.load('color_map.npy')

images_per_file = 64
n=0

f_selec = 0 #distinto contador topado en 64
#features = torch.load('/media/eric/Samsung_T5/stego_features/stego_features_train_extra_'+str((n+1)*images_per_file)+'.pt')['features']
features = torch.load('stego_features/stego_features_'+str((n+1)*64)+'.pt')['features']

n_files=8
for selec in  tqdm(range(images_per_file*n_files)):
    cimg = imageio.imread('reduced_cluster_v1/reduced_'+str(selec)+'.png')
    
    
    mask_color= np.stack((np.ones((320,320))*color_map[c][0],
                 np.ones((320,320))*color_map[c][1],
                 np.ones((320,320))*color_map[c][2]),axis=-1)

    mask0 = cimg[:,:,0] == mask_color[:,:,0] 
    mask1 = cimg[:,:,1] == mask_color[:,:,1] 
    mask2 = cimg[:,:,2] == mask_color[:,:,2]
    mask = mask0*mask1*mask2
    
         # coordenadas de los pixeles del cluster
    x,y = np.where(mask)
    # leemos el siguiente archivo
    if selec % images_per_file == 0 and selec!=0:
        n=n+1
        features = 0
        #features = torch.load('/media/eric/Samsung_T5/stego_features/stego_features_train_extra_'+str((n+1)*images_per_file)+'.pt')['features']
        # features = torch.load('/media/eric/SAMSUNG/stego_features_original_'+str((n+1)*64)+'.pt')['features']
        features = torch.load('stego_features/stego_features_'+str((n+1)*64)+'.pt')['features']
        f_selec = 0
    # si la imagen tiene pixeles del target cluster modificamos el color
    if len(x)!=0:
        X = np.array(features[f_selec][:,x,y].T)
        X = scaler.transform(X)
        pred = model.predict(X)
        
        #asignamos los colores
        for i in range(len(y)):
            cimg[x[i],y[i]] = custom_palette[pred[i]]
            
    im = Image.fromarray(cimg)
    im.save('reduced_cluster_v2/reduced_'+str(selec)+'.png')        
    f_selec = f_selec+1
      
        

    
    
    
    
    
    
    
    
    
    