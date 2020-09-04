import torch
import torch.nn as nn
import torch.nn.functional as F 




class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super (BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels)
    

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x)


class InceptionA(nn.Module):
    def __init__(self,in_channels,pool_features,conv_block=None):
        super(InceptionA,self).__init__()
        if conv_block is None:
            conv_block=BasicConv2d
        self.branch1x1=conv_block(in_channels,64,kernel_size=1)

        self.branch5x5_1=conv_block(in_channels,48,kernel_size=1)
        self.branch5x5_2=conv_block(48,64,kernel_size=5,padding=2)

        self.branch3x3dbl_1=conv_block(in_channels,64,kernel_size=1)
        self.branch3x3dbl_2=conv_block(64,96,kernel_size=3,padding=1)
        self.branch3x3dbl_3=conv_block(96,96,kernel_size=3,padding=1)

        self.branch_pool=conv_block(in_channels,pool_features,kernel_size=1)
    
        self.describe={'in_channels':in_channels,'pool_feature':pool_features,'conv_block':conv_block}
    

    def _forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3dbl=self.branch3x3dbl_1(x)
        branch3x3dbl=self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl=self.branch3x3dbl_3(branch3x3dbl)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch1x1,branch5x5,branch3x3dbl,branch_pool]
        return outputs
    
    def forward(self,x):
        outputs=self._forward(x)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return torch.cat(outputs,1)

class InceptionB(nn.Module):
    def __init__(self,in_channels,conv_block=None):
        super(InceptionB,self).__init__()
        if conv_block is None:
            conv_block=BasicConv2d
        self.branch3x3=conv_block(in_channels,384,kernel_size=3,stride=2)

        self.branch3x3dbl_1=conv_block(in_channels,64,kernel_size=1)
        self.branch3x3dbl_2=conv_block(64,96,kernel_size=3,padding=1)
        self.branch3x3dbl_3=conv_block(96,96,kernel_size=3,stride=2)
    
        self.describe={'in_channels':in_channels,'conv_block':conv_block}

    def _forward(self,x):
        branch3x3=self.branch3x3(x)

        branch3x3dbl=self.branch3x3dbl_1(x)
        branch3x3dbl=self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl=self.branch3x3dbl_3(branch3x3dbl)

        branch_pool=F.max_pool2d(x,kernel_size=3,stride=2)

        outputs=[branch3x3,branch3x3dbl,branch_pool]
        return outputs
        
    def forward(self,x):
        outputs=self.forward(x)
        return torch.cat(outputs,1)


class BottleNeck(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1):
        super(BottleNeck,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,self.expansion*out_channels,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
   
        self.describe={'in_channels':in_channels}
   
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn2(self.conv3(out)))
        return out

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(
            in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False
        )
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(
            out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False
        )
        self.bn2=nn.BatchNorm2d(out_channels)

        self.short_cut=nn.Sequential()
        if stride!=1 or in_channels!=self.expansion*out_channels:
            self.short_cut=nn.Sequential(
                nn.Conv2d(in_channels,self.expansion*out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.short_cut(x)
        out=F.relu(out)
        return out

class Resblock(nn.Module):
    def __init__(self,block,in_channels,out_channels,num_blocks,stride,num_classes=10):
        super(Resblock,self).__init__()
        #self.in_channels=64
        self.layer=self.make_layer(block,in_channels,out_channels,num_blocks,stride)

    def make_layer(self,block,in_channels,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(in_channels,out_channels,stride))
            self.in_channels=out_channels*block.expansion
            return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.layer(x)
        return out