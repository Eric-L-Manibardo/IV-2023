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



color_map = np.load('color_map_GT.npy')
n_classes=len(color_map)



n_items = 500
for selec in  tqdm(range(n_items)):

    cimg = imageio.imread('GT/gt_'+str(selec)+'.png')

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
        ##vehicle merges: car, truck, bus, caravan, and train   
        if c==1 or c==2 or c==3 or c==6 or c==7:
            group=4
        #road merges: road, parking and rail track    
        elif c==23 or c==26 or c==19:
            group=15
        #construction merges: building, wall, fence, bridge, tunnel
        elif c==11 or c==20 or c==16:
            group=8
        #traffic signs
        elif c==25:
            group=22
        # bike / motorbikes
        elif c==5:
            group=14
        # car
        # elif c==4: hay sombras de pedestrian
        #     group=20
            
        skeleton[mask] = color_map[group]
        
    im = Image.fromarray(skeleton.astype('uint8'))
    im.save('GT_v2/gt_'+str(selec)+'.png')
    

    






