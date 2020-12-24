import torch.nn as nn
import torch



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
            nn.ConvTranspose1d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.Conv1d(ngf * 16, ngf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 
            nn.ConvTranspose1d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.Conv1d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 
            nn.ConvTranspose1d( ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*4),
            nn.ReLU(True),
            nn.Conv1d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 
            nn.ConvTranspose1d( ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            nn.Conv1d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ng*2) x 64 
            nn.ConvTranspose1d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.Conv1d(ngf , ngf , 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf ),
            nn.ReLU(True),
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
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*2, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*4, ndf*4, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*8, ndf*8, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*16, ndf*16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
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
