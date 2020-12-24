import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init



class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        # self.bn=nn.LayerNorm(out_size)
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
            
    def swish(self,x):
        return x * F.sigmoid(x)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        # outputs=F.relu(outputs)
        outputs = F.leaky_relu(outputs,0.2)
        # outputs=self.swish(outputs)
        
        
        if self.dov>0:
            outputs = self.do(outputs)
        
        return outputs


class Discriminator(nn.Module):
    
    
    def __init__(self, levels=5,lvl1_size=3,input_size=12,output_size=1,convs_in_layer=2,init_conv=4,filter_size=3):
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        self.filter_size=filter_size
        
        
        
        self.init_conv=myConv(input_size,init_conv,filter_size=filter_size)
        
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            
            if lvl_num==0:
                self.layers.append(myConv(init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            else:
                self.layers.append(myConv(int(lvl1_size*(lvl_num))+int(lvl1_size*(lvl_num))+init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            
            for conv_num_in_lvl in range(self.convs_in_layer-1):
                self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size))


        self.conv_final=myConv(int(lvl1_size*(self.levels))+int(lvl1_size*(self.levels))+init_conv, int(lvl1_size*self.levels),filter_size=filter_size)
        
        self.fc=nn.Linear(int(lvl1_size*self.levels), self.output_size)
        
        
        
        
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
        
        
    def forward(self, x):
        
        
        
        x=self.init_conv(x)
        
        x0=x
        
        ## aply all convolutions
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            for conv_num_in_lvl in range(self.convs_in_layer):
                layer_num+=1
                if conv_num_in_lvl==1:
                    y=x
                
                x=self.layers[layer_num](x)
                
            ## skip conection to previous layer and to the input
            x=torch.cat((F.avg_pool1d(x0,2**lvl_num,2**lvl_num),x,y),1)
            
            x=F.max_pool1d(x, 2, 2)
            
            
            
        x=self.conv_final(x)
        
        
        x=F.adaptive_max_pool1d(x,1)
        
        
        # N,C,1
        
        x=x.view(list(x.size())[:2])
        
        # N,C
        
        x=self.fc(x)
        
        # x=torch.sigmoid(x)
        
        return x   