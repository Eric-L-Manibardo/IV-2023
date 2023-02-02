
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
from scipy.ndimage.measurements import label
from collections import Counter
import matplotlib.pyplot as plt

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


color_map = np.load('color_map_reduced_2.npy')


new_color_map = np.array([[ 0,   0, 0],#shadow
                 [  244,   35,   232],#sidewalk
                 [  1000,1000,1000], # non used(former bricks)
                 [  220, 220, 0], #traffic signs
                 [  0,  0, 142], # car wheels
                 [  107,  142, 35],# vegetation
                 [ 70, 130, 180], #sky
                 [ 128,  64,   128], #road
                 [  0,   0, 142], # vehicle
                 [  40,   40, 40], # static ego (ignore)
                 [  128, 64, 128], #railway -> road
                 [255,255,255], # noise detaills (ignore)
                 [  153, 153, 153], # vertical poles
                 [  220,  20,  60],# pedestrian
                 [  70,  70, 70],# building
                 [ 152, 251,  152],# h_vegetation
                 [  119,   11, 32]]) #bike

n_images=500
for selec in  tqdm(range(n_images)):
    cimg = imageio.imread('reduced_cluster_v9/reduced_'+str(selec)+'.png')
    skeleton = np.zeros(320*320*3).reshape(320,320,3)

    for c in range(len(color_map)):
        
        mask_color= np.stack((np.ones((320,320))*color_map[c][0],
                      np.ones((320,320))*color_map[c][1],
                      np.ones((320,320))*color_map[c][2]),axis=-1)
        mask0 = cimg[:,:,0] == mask_color[:,:,0] 
        mask1 = cimg[:,:,1] == mask_color[:,:,1] 
        mask2 = cimg[:,:,2] == mask_color[:,:,2]
        mask = mask0*mask1*mask2
            
        skeleton[mask] = new_color_map[c]
        
    im = Image.fromarray(skeleton.astype('uint8'))
    im.save('PREDICTION_custom/'+str(selec)+'.png')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    