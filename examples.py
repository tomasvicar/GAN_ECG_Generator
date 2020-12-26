import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from dataloader import DataLoader
import dcgan_1d
from datetime import datetime
import os


data_path = '../data_cinc2020_sinusonly'


workers = 0
batch_size = 64
latent_size = 100

loader = DataLoader(data_path)
loader = torch.utils.data.DataLoader(loader, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=True)

device = torch.device("cuda:0")


G = dcgan_1d.Generator(latent_size)
G.load_state_dict(torch.load('G000118.pt'))
G.eval()
G = G.to(device)



torch.manual_seed(60)


for i,data in enumerate(loader):

    G.train()
    
    
    x = data.to(device)

    z = torch.randn([batch_size,latent_size,1], device=device)
    
            
            

    Gz_fix = G(z).detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    
    fig, axs = plt.subplots(6, 1)
    fig.suptitle('fake')
    axs[0].plot(Gz_fix[0,0,:])
    axs[1].plot(Gz_fix[0,5,:])
    axs[2].plot(Gz_fix[1,0,:])
    axs[3].plot(Gz_fix[1,5,:])
    axs[4].plot(Gz_fix[2,0,:])
    axs[5].plot(Gz_fix[2,5,:])
    plt.savefig('fake.png' )
    plt.show()
    


    fig, axs = plt.subplots(6, 1)
    axs[0].plot(x[0,0,:])
    axs[1].plot(x[0,5,:])
    axs[2].plot(x[1,0,:])
    axs[3].plot(x[1,5,:])
    axs[4].plot(x[2,0,:])
    axs[5].plot(x[2,5,:])
    plt.savefig('real.png' )
    plt.show()
    
    if not os.path.exists('examples'):
        os.makedirs('examples')
    for k in range(20):
        fig, axs = plt.subplots(6, 1)
        axs[0].plot(Gz_fix[k,0,:])
        axs[1].plot(Gz_fix[k,2,:])
        axs[2].plot(Gz_fix[k,4,:])
        axs[3].plot(Gz_fix[k,6,:])
        axs[4].plot(Gz_fix[k,8,:])
        axs[5].plot(Gz_fix[k,10,:])
        plt.savefig('examples/fake_example' + str(k).zfill(5) + '.png' )
        plt.show()
        
    for k in range(20):
        fig, axs = plt.subplots(6, 1)
        axs[0].plot(x[k,0,:])
        axs[1].plot(x[k,2,:])
        axs[2].plot(x[k,4,:])
        axs[3].plot(x[k,6,:])
        axs[4].plot(x[k,8,:])
        axs[5].plot(x[k,10,:])
        plt.savefig('examples/real_example' + str(k).zfill(5) + '.png' )
        plt.show()  
        
        
    break