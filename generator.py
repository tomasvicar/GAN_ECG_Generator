import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

class unetConv1(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()
        
        self.do_batch=do_batch
    
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)



    def forward(self, inputs):
        outputs = self.conv(inputs)
        
        if self.do_batch:
            outputs = self.bn(outputs)          
        outputs=F.relu(outputs)

        return outputs
    
    
    

class unetConvT1(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_size, out_size,filter_size,stride=stride, padding=pad, output_padding=out_pad)
        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs
    
    
    

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        self.up = unetConvT1(in_size, out_size )
        


    def forward(self, inputs):
       


        return self.up(inputs)




class Generator(nn.Module):
    def __init__(self, filters=np.array([256,128,64,32,16]).astype(np.int),in_size = 12,out_size = 12):
        super().__init__()
        
        self.out_size = out_size
        self.in_size = in_size
        self.filters = filters
        
        self.init_conv = unetConv1(self.in_size , filters[0])
        
        
        self.up_conv_list =nn.ModuleList()
        self.conv_list =nn.ModuleList()
        for k in range(len(self.filters)-1):
            self.conv_list.append(nn.Sequential(unetConv1(filters[k], filters[k]),unetConv1(filters[k], filters[k] )))
            self.up_conv_list.append(unetUp(filters[k], filters[k+1] ))
    
    
        self.final_conv = nn.Sequential(unetConv1(filters[-1], filters[-1]),unetConv1(filters[-1], filters[-1] ))
        
        self.final_conv_11 = nn.Conv1d(filters[-1],self.out_size,1)
        
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            
        
        
        
    def forward(self, x):
        
        x = self.init_conv(x)
        
        for k in range(len(self.filters)-1):
            x = self.conv_list[k](x)
            x = self.up_conv_list[k](x)
        
        x = self.final_conv(x)
        x = self.final_conv_11(x)
        
        return x