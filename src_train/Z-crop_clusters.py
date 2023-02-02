
from PIL import Image
import numpy as np
import imageio
from tqdm import tqdm


# d=torch.load("results_stego.pt")

# #unpack dicts
# images=d["image"]
# cluster=d["cluster"]

selec=0
color_map = np.load('color_map.npy')
n_classes=len(color_map)





for selec in tqdm(range(2975)):
    img = imageio.imread('images/image'+str(selec)+'.png')
    cimg = imageio.imread('cluster/cluster'+str(selec)+'.png')

    for c in range(n_classes):
    
        mask_color= np.stack((np.ones((320,320))*color_map[c][0],
                     np.ones((320,320))*color_map[c][1],
                     np.ones((320,320))*color_map[c][2]),axis=-1)
        mask0 = cimg[:,:,0] == mask_color[:,:,0] 
        mask1 = cimg[:,:,1] == mask_color[:,:,1] 
        mask2 = cimg[:,:,2] == mask_color[:,:,2]
        mask = mask0*mask1*mask2
        result = img*np.stack((mask,mask,mask),axis=-1)
        
        im = Image.fromarray(result)
        im.save('masked_clusters/class_'+str(c)+'/mask_'+str(selec)+'.png')
    







