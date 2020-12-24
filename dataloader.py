import glob
import os
from skimage.io import imread, imsave
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class DataLoader(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path=path

        self.files_img=glob.glob(self.path + '/**/*.npy', recursive=True)

        self.num_of_imgs=len(self.files_img)

    def __len__(self):
        return self.num_of_imgs


    def __getitem__(self, index):
        
        data = np.load(self.files_img[index])

        data = torch.Tensor(data.astype(np.float32))
        
    
        return data
    
    
    
    
    
    
if __name__ == "__main__": 
    loader = DataLoader('../data_64')
    loader = torch.utils.data.DataLoader(loader, batch_size=128,shuffle=True, num_workers=0,drop_last=True)
    
    for i,data in enumerate(loader):
        
        plt.figure(figsize=(15,15))
        img = np.transpose(vutils.make_grid(data.cpu().detach()[:32],padding=2, normalize=True).numpy(),(1,2,0))
        plt.imshow(img)
        plt.show()
        imsave('example_img/real' + str(i).zfill(7) + '.png',img)
    
        if i ==3: 
            break