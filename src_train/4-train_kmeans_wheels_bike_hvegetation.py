#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:50:45 2022

@author: eric
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

'''
1 - pixel_per_single_cluster.py
2 - features_per_pixel.py
3 - train_kmeans

'''
# =============================================================================
# partimos en 3 para acelerar al append, que se ralentiza con el tamaño
# =============================================================================
c=16 #cluster for bike, wheels, ground vegetation
all_features = np.empty((0,100),'float32') #vacio con 100 features
n_items=47
for selec in tqdm(range(n_items)):
    with open('features_per_pixel_cluster_'+str(c)+'/features_cluster_'+str(c)+'_part_'+str(selec)+'.pkl', 'rb') as f:
        a = pickle.load(f)
    all_features = np.append(all_features, a, axis=0)

#ultimo archivo
#with open('features_per_pixel_cluster_'+str(c)+'/features_cluster_'+str(c)+'_part_end.pkl', 'rb') as f:
 #       a = pickle.load(f)
#all_features = np.append(all_features, a, axis=0)


print('All features collected!!')
scaler = MinMaxScaler()
scaler.fit(all_features)
X = scaler.transform(all_features)

print('Training K-means....')
batch_size = int(len(X)/10)
model = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=batch_size).fit(X)

# 
with open('kmeans_models/kmeans_'+str(c)+'.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('kmeans_models/scaler_'+str(c)+'.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Model and scaler stored!')

pred = model.predict(X)
count_arr = np.bincount(pred)
print('Occurrences of clusters:'+str(count_arr/len(pred)))
