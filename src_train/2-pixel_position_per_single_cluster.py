#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:48:41 2022

@author: eric
"""

import numpy as np
import imageio
from tqdm import tqdm
import pickle


color_map = np.load('color_map.npy')


c=16
color_target = color_map[c]
# creamos un dict 
dx,dy=dict(),dict()
n_items = 2975
for selec in  tqdm(range(n_items)):

    cimg = imageio.imread('reduced_cluster_v1/reduced_'+str(selec)+'.png')
    
    
    mask_color= np.stack((np.ones((320,320))*color_target[0],
                  np.ones((320,320))*color_target[1],
                  np.ones((320,320))*color_target[2]),axis=-1)
    mask0 = cimg[:,:,0] == mask_color[:,:,0] 
    mask1 = cimg[:,:,1] == mask_color[:,:,1] 
    mask2 = cimg[:,:,2] == mask_color[:,:,2]
    mask = mask0*mask1*mask2
    
    # coordenadas de los pixeles del cluster
    x,y = np.where(mask)

    dx[selec]=x
    dy[selec]=y



dictionary = {'x':dx,'y':dy}
with open('pixel_pos/pixel_pos_per_cluster_'+str(c)+'.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

