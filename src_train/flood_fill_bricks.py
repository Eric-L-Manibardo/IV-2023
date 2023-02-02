
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
from skimage.segmentation import flood, flood_fill
from scipy.ndimage.measurements import label
from collections import Counter
import matplotlib.pyplot as plt

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


color_map = np.load('color_map_reduced_v3.npy')

color_target = np.array([0,196,156]) #azul cyan bricks
id_target = np.where(np.equal(color_map, color_target).all(axis=1))[0][0]

color_ego = np.array([106,156,156])
id_ego = np.where(np.equal(color_map, color_ego).all(axis=1))[0][0]

color_wheel = np.array([100,0,0])
id_wheel = np.where(np.equal(color_map, color_wheel).all(axis=1))[0][0]

color_car = np.array([[128, 192, 128]])
id_car = np.where(np.equal(color_map, color_car).all(axis=1))[0][0]

color_shadow = np.array([[0, 0, 186]])
id_shadow = np.where(np.equal(color_map, color_shadow).all(axis=1))[0][0]

color_road = np.array([[76, 91, 76]])
id_road = np.where(np.equal(color_map, color_road).all(axis=1))[0][0]

color_sidewalk = np.array([0,0,166])

for selec in range(100):    
    # selec = 142

    # =============================================================================
    # translate RGB to int matrix
    # =============================================================================
    cimg = imageio.imread('reduced_cluster_v5/reduced_'+str(selec)+'.png')
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
    
        lmask = labeled_mask == k + 1 #label 0 es todo lo que no son islas
        x,y = np.where(lmask)
        
        
        '''
        el algortimo coge las x, ya que estan en contacto con el 5 (target)
        
        111111111111111111111111111111111111111111111111111111111111111111111
        111111111111111111111111111111111111111111111111111111111111111111111
        111111111111111111111111111111111111111111111111111111111111111111111
        111111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxx111111111111111111111111111
        111111111111111x5555555555555555555555555x111111111111111111111111111
        11111111111111x5555555555555555555555xxxxx111111111111111111111111111
        11111111111111xx5555555555xxxxxxxxxxxx1111111111111111111111111111111
        111111111111111xxxxxxxxxxxx111111111111111111111111111111111111111111
        111111111111111111111111111111111111111111111111111111111111111111111
        
        añadimos un -1 si el pixel está fuera de imagen
        '''
        neighbor_pixels, border = list(),list()
        #por cada fila
        for i in range(len(np.unique(x))):
            row = np.where(x==np.unique(x)[i])[0]
            #primer y ultima columna de la fila i
            first = row[0]
            last  = row[-1]
            #recogemos todos los valores de la fila superior    
            for j in range(len(row)):
                try:            
                    neighbor_pixels.append(skeleton[np.unique(x)[i]-1, y[row[j]]])
                except:
                    neighbor_pixels.append(-1)
                    # pass
            #añadimos en diagonal del primero y del ultimo
            try:
                neighbor_pixels.append(skeleton[np.unique(x)[i]-1, y[row[0]]-1])
            except:
                neighbor_pixels.append(-1)
                # pass
            try:
                neighbor_pixels.append(skeleton[np.unique(x)[i]-1, y[row[j]]+1])
            except:
                neighbor_pixels.append(-1)
                # pass
        
        #finalmente añadimos los pixeles de la fila de abajo
        for j in range(len(row)):
            try:
                neighbor_pixels.append(skeleton[np.unique(x)[i]+1, y[row[j]]])
            except:
                neighbor_pixels.append(-1)
                # pass
        #añadimos en diagonal del primero y del ultimo
        try:
            neighbor_pixels.append(skeleton[np.unique(x)[i]+1, y[row[0]]-1])
        except:
            neighbor_pixels.append(-1)
            # pass
        try:
            neighbor_pixels.append(skeleton[np.unique(x)[i]+1, y[row[j]]+1])
        except:
            neighbor_pixels.append(-1)
            # pass
        
        # eliminamos los pixeles de la propia isla
        neighbor_pixels = remove_values_from_list(neighbor_pixels, id_target)
        d_count=Counter(neighbor_pixels)  
        
# =============================================================================
# contamos los pixeles de cada label
# =============================================================================
      
        if d_count[-1]+d_count[id_wheel]+d_count[id_car]+d_count[id_shadow]+d_count[id_road]+d_count[id_ego] >= 0.50 * len(neighbor_pixels):
            key_colors.append(color_road)
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
    im.save('reduced_cluster_v6/reduced_'+str(selec)+'.png')
        
    # break

    
    
    
    
    
    
    
    
    
    
    
    
    
    