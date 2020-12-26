import torch.nn as nn
import torch


class GenLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        self.tconv = nn.Sequential(
            nn.ConvTranspose1d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            )
        self.convs1 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
        )
        self.convs2 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
        )
        self.convs3 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
        )
        self.convs4 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
        )
        self.convs5 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_size),
            nn.ReLU(True),
        )
        
        
    def forward(self, x):
        x = self.tconv(x)
        y1 = self.convs1(x)
        y2 = self.convs2(x+y1)
        y3 = self.convs3(x+y1+y2)
        y4 = self.convs4(x+y1+y2+y3)
        y5 = self.convs5(x+y1+y2+y3+y4)
        outputs = x+y1+y2+y3+y4+y5
        return outputs


class Generator(nn.Module):
    def __init__(self,latent_size):
        nz = latent_size
        ngf = 32
        nc = 12
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d( nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 
            GenLayer(ngf * 32, ngf * 16),
            # state size. (ngf*16) x 8 
            GenLayer(ngf * 16, ngf * 8),
            # state size. (ngf*8) x 16 
            GenLayer(ngf * 8, ngf * 4),
            # state size. (ngf*4) x 32 
            GenLayer(ngf * 4, ngf * 2),
            # state size. (ng*2) x 64 
            GenLayer(ngf * 2, ngf * 1),
            # state size. (ngf) x 128 
            nn.ConvTranspose1d( ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.Conv1d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 
        )
        
        self.apply(self.weights_init)

    def forward(self, input):
        return self.main(input)*2
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight)

    

class DisLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        G = 32
        
        self.sconv = nn.Sequential(
            nn.Conv1d(in_size, out_size, 4, 2, 1, bias=False),
            nn.GroupNorm(G,out_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convs1 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(G,out_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(G,out_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(G,out_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convs2 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
        )
        self.convs3 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
        )
        self.convs4 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
        )
        self.convs5 = nn.Sequential(
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
            nn.Conv1d(out_size, out_size, 3, 1, 1, bias=False),
            nn.GroupNorm(int(out_size/16),out_size),
            nn.LeakyReLU(G, inplace=True),
        )
        
    def forward(self, x):
        x = self.sconv(x)
        y1 = self.convs1(x)
        y2 = self.convs2(x+y1)
        y3 = self.convs3(x+y1+y2)
        y4 = self.convs4(x+y1+y2+y3)
        y5 = self.convs5(x+y1+y2+y3+y4)
        outputs = x+y1+y2+y3+y4+y5
        return outputs  
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 12
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256 
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128
            DisLayer(ndf * 1, ndf * 2),
            # state size. (ndf*2) x 64
            DisLayer(ndf * 2, ndf * 4),
            # state size. (ndf*4) x 32
            DisLayer(ndf * 4, ndf * 8),
            # state size. (ndf*8) x 16
            DisLayer(ndf * 8, ndf * 16),
            # state size. (ndf*16) x 8
            nn.Conv1d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4
            nn.Conv1d(ndf * 32, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        
        self.apply(self.weights_init)

    def forward(self, input):
        return self.main(input)
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight)
