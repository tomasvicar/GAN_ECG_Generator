import glob
import os
from skimage.io import imread
import numpy as np

path = '../res_12_26_20__08_24'

names = glob.glob(path + '/fake*.png')



# import imageio
# with imageio.get_writer('/content/drive/MyDrive/gan_res.gif', mode='I') as writer:
#     for i,filename in enumerate(names):
#         print(i)
#         image = imageio.imread(filename)
#         writer.append_data(image)




import imageio

images = []
for i,file in enumerate(names[:119]):
    print(i)
    images.append(imageio.imread(file))


imageio.mimwrite('gan_res2.gif', images, fps=3)