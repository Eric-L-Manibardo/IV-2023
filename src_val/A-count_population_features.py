#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:51:00 2022

@author: eric
"""
import pickle
import pandas as pd
from tqdm import  tqdm
import numpy as np

n_clusters=13
d = dict()
for i in tqdm(range(n_clusters)):
    count = 0
    for part in range(86): #number of files
        with open('features_per_pixel_final/fpp_c'+str(i)+'/features_cluster_'+str(i)+'_part_'+str(part)+'.pkl', 'rb') as f:
            features = pickle.load(f)
        count = count + len(features)
    d[i] = count


with open('population_A.pkl', 'wb') as handle:
    pickle.dump(d, handle)
