#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:58:15 2022

@author: eric
"""



import numpy as np
import imageio
from tqdm import tqdm

color=list()
n_items = 2975
for i in tqdm(range(n_items)):
    cimg = imageio.imread('PREDICTION_custom/'+str(i)+'.png')
    color.append(np.unique(cimg.reshape(-1, cimg.shape[2]), axis=0))
# list 
'''At first, we flatten rows and columns of matrix. 
Now the matrix has as much rows as there're pixels in the image.
Columns are color components of each pixels'''


print(np.unique(np.vstack(color),axis=0))

np.save('color_map_custom',np.unique(np.vstack(color),axis=0))




