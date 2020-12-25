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
    loader = DataLoader('../data_cinc2020_sinusonly')
    loader = torch.utils.data.DataLoader(loader, batch_size=16,shuffle=True, num_workers=0,drop_last=True)
    
    for i,x in enumerate(loader):
        
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(x[0,0,:])
        axs[1].plot(x[0,5,:])
        axs[2].plot(x[1,0,:])
        axs[3].plot(x[1,5,:])
        plt.show()
        break    