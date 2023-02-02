#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:37:55 2022

@author: eric
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:41:46 2022

@author: eric
"""

from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm



color_map = np.load('color_map_linear.npy')
n_classes=len(color_map)



n_items = 500
for selec in  tqdm(range(n_items)):

    cimg = imageio.imread('linear/linear'+str(selec)+'.png')

    skeleton = np.zeros(320*320*3).reshape(320,320,3)

    for c in range(n_classes):
        group = c
        mask_color= np.stack((np.ones((320,320))*color_map[c][0],
                      np.ones((320,320))*color_map[c][1],
                      np.ones((320,320))*color_map[c][2]),axis=-1)
        mask0 = cimg[:,:,0] == mask_color[:,:,0] 
        mask1 = cimg[:,:,1] == mask_color[:,:,1] 
        mask2 = cimg[:,:,2] == mask_color[:,:,2]
        mask = mask0*mask1*mask2
        '''
        Cluster identifiers are extracted from a list from previous script
        '''

        #vehicle merges: car, truck, bus, caravan, and train  
        if c==0 or c==1 or c==4 or c==5:
            group=2
        #road merges: road, parking and rail track  
        elif c==19 or c==22:
            group=11
        #construction merges: building, wall, fence, bridge, tunnel 
        elif c==8 or c==16 or c==12 or c==13:
            group=6
        #traffic signs with lights
        elif c==21:
            group=18
        # motorcycle with bicycle
        elif c==3:
            group=10
        # car
        # elif c==4: hay sombras de pedestrian
        #     group=20
        # elif c==22:
        #     group=17#esto mergea rider con pedestrian
            
        skeleton[mask] = color_map[group]
        
    im = Image.fromarray(skeleton.astype('uint8'))
    im.save('PREDICTION_linear_v3/linear_'+str(selec)+'.png')
    

    






