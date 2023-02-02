
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
from tqdm import tqdm

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


color_map = np.load('color_map_reduced_2.npy')

color_target = np.array([150,120,90]) #railways o border
id_target = np.where(np.equal(color_map, color_target).all(axis=1))[0][0]

color_ego = np.array([150,100,100])
id_ego = np.where(np.equal(color_map, color_ego).all(axis=1))[0][0]

color_wheel = np.array([100,0,0])
id_wheel = np.where(np.equal(color_map, color_wheel).all(axis=1))[0][0]

color_car = np.array([[128, 64, 128]])
id_car = np.where(np.equal(color_map, color_car).all(axis=1))[0][0]

color_shadow = np.array([[0, 0, 70]])
id_shadow = np.where(np.equal(color_map, color_shadow).all(axis=1))[0][0]

color_road = np.array([[119, 11, 32]])
id_road = np.where(np.equal(color_map, color_road).all(axis=1))[0][0]

color_sidewalk = np.array([0,0,90])

resolution = 320
n_images=500
for selec in tqdm(range(n_images)):    
    # selec = 142

    # =============================================================================
    # translate RGB to int matrix
    # =============================================================================
    cimg = imageio.imread('reduced_cluster_v8/reduced_'+str(selec)+'.png')
    final_img = cimg.copy()
    skeleton = np.zeros(320*320).reshape(320,320)
    
    for c in range(len(color_map)):
        mask_color= np.stack((np.ones((320,320))*color_map[c][0],
                  np.ones((320,320))*color_map[c][1],
                  np.ones((320,320))*color_map[c][2]),axis=-1)
        mask0 = cimg[:,:,0] == mask_color[:,:,0] 
        mask1 = cimg[:,:,1] == mask_color[:,:,1] 
        mask2 = cimg[:,:,2] == mask_color[:,:,2]
        mask = mask0*mask1*mask2
        skeleton[mask]=c
    
    # =============================================================================
    # generamos la mascara target
    # =============================================================================
    mask_color= np.stack((np.ones((320,320))*color_target[0],
              np.ones((320,320))*color_target[1],
              np.ones((320,320))*color_target[2]),axis=-1)
    mask0 = cimg[:,:,0] == mask_color[:,:,0] 
    mask1 = cimg[:,:,1] == mask_color[:,:,1] 
    mask2 = cimg[:,:,2] == mask_color[:,:,2]
    mask = mask0*mask1*mask2
    
    # separamos en islas los pixeles del mismo color
    labeled_mask, n_islands = label(mask)
    key_colors = list()
    for k in range(n_islands):
    
        #filtramos la isla en cuestión
        lmask = labeled_mask == k + 1 #label 0 es todo lo que no son islas
        x,y = np.where(lmask) #coordenadas de los pixeles de la isla en cuestión
        
        '''
        vamos a cambiar el algoritmo. Para cada pixel de cada isla,
        se anota los pixeles vecinos de cada pixel.
        if not (RGB_Vecino == target) and not already_saved:
            save position (i.e. (x,y))

        Cuando tengamos todas las posiciones, devolvemos el valor de la maskara en si
        
        añadimos un -1 si el pixel está fuera de imagen en un array especial, 
        para tenerlo en cuenta en los casos pertinentes 
        '''
        neigh_pos_x = [1,1,1, 0,0, -1,-1,-1]
        neigh_pos_y = [-1,0,1,-1,1, -1,0,1 ]
        neighbor_pixels, out_pixels = list(),list()
        #para cada pixel
        for i in range(len(x)):
            for p in range(len(neigh_pos_x)):                
                neigh = (x[i]+neigh_pos_x[p], y[i]+neigh_pos_y[p])
                # si el vecino está en la imagen
                if neigh[0]>0 and neigh[1]>0 and neigh[0]<resolution and neigh[1]<resolution:
                    pxl = skeleton[neigh[0], neigh[1]]
                    if  pxl != id_target and neigh not in neighbor_pixels:
                        neighbor_pixels.append(neigh)
                #fuera de imagen
                else:
                    if neigh not in out_pixels:
                        out_pixels.append(neigh)
        
        
# =============================================================================
# contamos los pixeles de cada label
# =============================================================================
        neigh_labels = list()
        for i in range(len(neighbor_pixels)):
            neigh_labels.append(skeleton[neighbor_pixels[i][0], neighbor_pixels[i][1]])
        for i in range(len(out_pixels)):
            neigh_labels.append(-1)
        d_count=Counter(neigh_labels)  
      
        if d_count[-1]+d_count[id_wheel]+d_count[id_car]+d_count[id_shadow]+d_count[id_road]+d_count[id_ego] >= 0.90 * len(neigh_labels):
            key_colors.append(color_target)
        else:
            key_colors.append(color_sidewalk)
    
        
    
    # =============================================================================
    # cambio el color de cada isla despues de las mascaras
    # =============================================================================
    
    
    
    # # =============================================================================
    # # save result
    # # =============================================================================

    
    for k in range(len(key_colors)):
        lmask = labeled_mask == k+1
        final_img[lmask] = key_colors[k] 
    im = Image.fromarray(final_img)
    im.save('reduced_cluster_v9/reduced_'+str(selec)+'.png')
        
    # break

    
    
    
    
    
    
    
    
    
    
    
    
    
    