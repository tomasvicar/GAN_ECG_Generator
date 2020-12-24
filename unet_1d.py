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
        


    def forward(self, inputs1, inputs2):
       
        inputs2 = self.up(inputs2)


        return torch.cat([inputs1, inputs2], 1)




class Unet1d(nn.Module):
    def __init__(self, filters=(np.array([8,16,32,64,128])).astype(np.int),in_size=12,out_size=12):
        super().__init__()
        
        self.out_size = out_size
        
        self.in_size = in_size
        
        self.filters = filters
        
        
        self.conv1 = nn.Sequential(unetConv1(in_size, filters[0]),unetConv1(filters[0], filters[0]),unetConv1(filters[0], filters[0]))

        
        self.conv_list =nn.ModuleList()
        for k in range(1,len(self.filters)-1):
            self.conv_list.append(nn.Sequential(unetConv1(filters[k-1], filters[k] ),unetConv1(filters[k], filters[k] ),unetConv1(filters[k], filters[k] )))
        

        self.center = nn.Sequential(unetConv1(filters[-2], filters[-1] ),unetConv1(filters[-1], filters[-1] ))
        
        
        self.up_concat_list =nn.ModuleList()
        self.up_conv_list =nn.ModuleList()
        for k in range(len(self.filters)-1,1,-1):
            self.up_concat_list.append(unetUp(filters[k], filters[k] ))
            self.up_conv_list.append(nn.Sequential(unetConv1(filters[k-1]+filters[k], filters[k-1] ),unetConv1(filters[k-1], filters[k-1] )))
        

        self.up_concat1 = unetUp(filters[1], filters[1])
        self.up_conv1=nn.Sequential(unetConv1(filters[0]+filters[1], filters[0] ),unetConv1(filters[0], filters[0],do_batch=0 ))
        
        self.final = nn.Conv1d(filters[0], self.out_size, 1)
        

        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            
        
        
        
    def forward(self, inputs):
        

        conv1 = self.conv1(inputs)
        x = F.max_pool1d(conv1,2,2)
        
        conv_outputs = []
        for k,conv in enumerate(self.conv_list):
            y = conv(x)
            conv_outputs.append(y)
            x = F.max_pool1d(y,2,2)
           

        x = self.center(x)

        for k,(up_concat,up_conv) in enumerate(zip(self.up_concat_list,self.up_conv_list)):
            conv_output = conv_outputs[-(k+1)]
            x = up_concat(conv_output, x)
            x = up_conv(x)
           
            
            
        
        x = self.up_concat1(conv1, x)
        x=self.up_conv1(x)
        
        x = self.final(x)
        
        return x