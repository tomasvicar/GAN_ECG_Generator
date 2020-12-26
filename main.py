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


if __name__ == "__main__":
    
    data_path = '../data_cinc2020_sinusonly'
    save_folder = '../res_' + datetime.now().strftime("%D__%H_%M").replace('/','_')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    
    workers = 4
    batch_size = 64
    D_iter = 5
    latent_size = 100
    milestones = [80]
    

    lam = 10 
    alpha = 0.001
    beta_1 = 0
    beta_2 = 0.9
    
    
    
    device = torch.device("cuda:0")

    
    loader = DataLoader(data_path)
    loader = torch.utils.data.DataLoader(loader, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=True)
    
    
    G = dcgan_1d.Generator(latent_size)
    D = dcgan_1d.Discriminator()
    
    G = G.to(device)
    D = D.to(device)
        
    
    optimizerD = optim.Adam(D.parameters(), lr=alpha, betas=(beta_1, beta_2))
    optimizerG = optim.Adam(G.parameters(), lr=alpha, betas=(beta_1, beta_2))
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones, gamma=0.1)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones, gamma=0.1)
    
    G_losses = []
    D_losses = []

    
    iters =-1
    for epoch in range(milestones[-1]):
        G_losses_tmp = []
        D_losses_tmp = []
        for i,data in enumerate(loader):
            iters = iters +1
            D.train()
            G.train()
            
            
            x = data.to(device)
    
            z = torch.randn([batch_size,latent_size,1], device=device)
            if iters == 0:
                z_fix = z
                x_fix = x.detach().cpu().numpy()
            
            Gz = G(z).detach()
               
            Dx = D(x)
            
            DGz = D(Gz)
            
            
            eps = torch.rand(batch_size, 1, device=device)
            eps = eps.expand(batch_size, int(x.nelement()/batch_size)).contiguous().view(x.size())
            interpolates = eps * x + ((1 - eps) * Gz)
            interpolates=interpolates.detach()
            interpolates.requires_grad=True
            out_interpolates=D(interpolates)
            gradients = autograd.grad(outputs=out_interpolates, inputs=interpolates,grad_outputs=torch.ones(out_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)*lam
            gradient_penalty=gradient_penalty.mean()
    

            loss_D =torch.mean(DGz) - torch.mean(Dx) + gradient_penalty
            
            
            D.zero_grad()
            loss_D.backward()
            optimizerD.step()
            
            
        
            D_losses_tmp.append(loss_D.cpu().detach().numpy())
            
            
            
            if (iters%D_iter) ==0:
                
                z = torch.randn([batch_size,latent_size,1], device=device)
                
                Gz = G(z)
                
                DGz = D(Gz)
                

                loss_G = -torch.mean(DGz)
                
                G.zero_grad()
                loss_G.backward()
                optimizerG.step()
            
            
                G_losses_tmp.append(loss_G.cpu().detach().numpy())
            
            
        schedulerD.step()
        schedulerG.step()
            
        D_losses.append(np.mean(D_losses_tmp))
        G_losses.append(np.mean(G_losses_tmp))
            
        # print("\014")
        plt.plot(D_losses)
        plt.title('D')
        plt.savefig(save_folder + '/Dloss' + str(epoch).zfill(6) + '.png' )
        plt.show()
        plt.plot(G_losses)
        plt.title('G')
        plt.savefig(save_folder + '/Gloss' + str(epoch).zfill(6) + '.png' )
        plt.show()
                
                
        D.eval()
        G.eval()
        Gz_fix = G(z_fix).detach().cpu().numpy()
        
        
        fig, axs = plt.subplots(6, 1)
        fig.suptitle('epoch ' + str(epoch))
        axs[0].plot(Gz_fix[0,0,:])
        axs[1].plot(Gz_fix[0,5,:])
        axs[2].plot(Gz_fix[1,0,:])
        axs[3].plot(Gz_fix[1,5,:])
        axs[4].plot(Gz_fix[2,0,:])
        axs[5].plot(Gz_fix[2,5,:])
        torch.save(G.state_dict(), save_folder + '/G' + str(epoch).zfill(6) + '.pt' )
        torch.save(D.state_dict(), save_folder + '/D' + str(epoch).zfill(6) + '.pt' )
        plt.savefig(save_folder + '/fake' + str(epoch).zfill(6) + '.png' )
        plt.show()
        
        



        # fig, axs = plt.subplots(4, 1)
        # axs[0].plot(x_fix[0,0,:])
        # axs[1].plot(x_fix[0,5,:])
        # axs[2].plot(x_fix[1,0,:])
        # axs[3].plot(x_fix[1,5,:])
        # plt.show()
        
        
        
                