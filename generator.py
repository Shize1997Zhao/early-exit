from random import choice
import torch
import torch.nn as nn
from blocks import *
import copy


blocks_list=['conv','inceptiona','inceptionb','bottleneck']

def module_create(mdef):

    modules = nn.Sequential()


    if mdef['type_name'] == 'conv':
        bn=mdef['describe']['batch_normalize']
        modules.add_module('Conv2d',nn.Conv2d(in_channels=mdef['describe']['in_ch'],
                           out_channels=mdef['describe']['out_ch'],
                           kernel_size=mdef['describe']['kernel_size'],
                           stride=mdef['describe']['stride'],
                           padding=mdef['describe']['kernel_size']//2,
                           groups=mdef['groups'] if 'groups' in mdef else 1,
                           bias= not bn))
        if bn:
            modules.add_module('BatchNorm2d',nn.BatchNorm2d(mdef['describe']['out_ch'],momentum=0.03,eps=1e-4))
        
        if mdef['describe']['activation']=='leaky':
            modules.add_module('activation',nn.LeakyReLU(0.1,inplace=True))
        elif mdef['describe']['activation']=='relu':
            modules.add_module('activation',nn.ReLU(inplace=False))


    if mdef['type_name']=='inceptiona':
        modules.add_module('InceptionA',InceptionA(in_channels=mdef['describe']['in_ch'],
                           pool_features=mdef['describe']['po_ft']))
    
    if mdef['type_name']=='inceptionb':
        modules.add_module('InceptionB',InceptionB(in_channels=mdef['describe']['in_ch'],
                            conv_block=mdef['describe']['conb']))
    
    if mdef['type_name']=='bottleneck':
        modules.add_module('BottleNeck',BottleNeck(in_channels=mdef['describe']['in_ch'],out_channels=mdef['describe']['out_ch']))

    if mdef['type_name']=='resblock':
        modules.add_module('resblock',Resblock(block=mdef['describe']['block'],
                           in_channels=mdef['describe']['in_ch'],
                           out_channels=mdef['describe']['out_ch'],
                           num_blocks=mdef['describe']['num_blocks'],
                           stride=mdef['describe']['stride']))
    
    
    if mdef['type_name']=='fc':
        modules.add_module('fc',nn.Linear(mdef['describe']['input_nums'],mdef['describe']['num_class']))
    
    return modules

class Net_generator(nn.Module):
    def __init__(self,tree_def_list,branch_node):
        super(Net_generator,self).__init__()
        self.branches=[]
        self.result_list=[]
        self.featuremap=[]
        self.tree_def_list=tree_def_list
        self.swap=None
        self.branch_node=branch_node
        self.modules_list=nn.ModuleList([])
        for mdef in (self.tree_def_list):
            self.modules_list+=[module_create(mdef)]
        self.all_modules_list = self.modules_list
        self.branches,self.next_tree_list=self.layers_make(self.tree_def_list)
        self.branch1=self.branches[0]
        self.branches,self.next_tree_list_=self.layers_make(self.next_tree_list)
        self.branch2=self.branches[0]
        self.linear1=nn.Linear(8192,10)
        self.linear2=nn.Linear(360448//pow(4,branch_node-1),10)
    
    def modules_make(self,tree_def_list):
        modules_list=[]
        for mdef in (tree_def_list):
            modules_list+=[module_create(mdef)]
        return modules_list


    def layers_make(self,tree_def_list):
        branch=[]                 
        layers=[]
        record=copy.deepcopy(tree_def_list)                           
        record.reverse()
        module_list=self.modules_make(tree_def_list)
        length=len(tree_def_list)
        for i in range(length):
            if i>0:
               record.pop()
               location=tree_def_list[i]['parent']-index
               layer=module_list[location]  
               layers.append(layer)
               if tree_def_list[i]['type_name']=='fc':
                   module=nn.Sequential(*layers)                           
                   branch.append(module)
                   record.pop()
                   break
            else:
                index=tree_def_list[i]['index']
        record.reverse()
        return branch,record




    def forward(self,x):
        outputs=[]
        featuremap_branch=None
        self.swap=x
        for i,module in enumerate(self.all_modules_list):
            if i>0:
                parent=self.featuremap[self.tree_def_list[i]['parent']]
                if self.tree_def_list[i]['type_name']=='fc':
                    parent=parent.view(parent.size(0),-1)
                self.swap=module(parent)
                if i == self.branch_node:
                    break
            else:
                self.swap=module(self.swap)
            self.featuremap+=[self.swap]
        featuremap_branch=self.swap
        self.featuremap=[]
        out=self.branch1(x)
        out=out.view(out.size(0),-1)
        out=self.linear1(out)
        outputs.append(out)
        out=self.branch2(featuremap_branch)
        out=out.view(out.size(0),-1)
        out=self.linear2(out)
        outputs.append(out)
        return outputs

   


def mdef_dict(index,type_name,describe,left_child,right_child,parent):
    mdef= {'index':index, 'type_name':type_name, 'describe':describe, 'left_child':left_child, 'right_child':right_child, 'parent':parent}
    return mdef




def sample_net_tree(branch_node):
    tree_def_list=[]
    tree_def_list+=[mdef_dict(0,'conv',{'kernel_size':3,'in_ch':3,'out_ch':64,'stride':1,'batch_normalize':True,'activation':'relu'},1,-1,-1)]  
    tree_def_list+=[mdef_dict(1,'resblock',{'block':BasicBlock,'in_ch':64,'out_ch':64,'num_blocks':3,'stride':1},2,5,0)]    #360448  90112     1
    tree_def_list+=[mdef_dict(2,'resblock',{'block':BasicBlock,'in_ch':64,'out_ch':128,'num_blocks':3,'stride':2},3,-1,1)]  #90112   正确      1/4
    tree_def_list+=[mdef_dict(3,'resblock',{'block':BasicBlock,'in_ch':128,'out_ch':256,'num_blocks':3,'stride':2},4,-1,2)] #22528   90112     1/16
    tree_def_list+=[mdef_dict(4,'resblock',{'block':BasicBlock,'in_ch':256,'out_ch':512,'num_blocks':3,'stride':2},8,-1,3)]
    tree_def_list+=[mdef_dict(5,'fc',{'input_nums':8192,'num_class':10},-1,-1,4)]
    tree_def_list+=[mdef_dict(6,'inceptiona',{'in_ch':tree_def_list[branch_node]['describe']['out_ch'],'po_ft':64},6,-1,branch_node)]
    tree_def_list+=[mdef_dict(7,'inceptiona',{'in_ch':288,'po_ft':128},7,-1,6)]
    #tree_def_list+=[mdef_dict(7,'bottleneck',{'in_ch':352,'out_ch':64},8,-1,2)]
    print(tree_def_list[branch_node]['parent'])
    tree_def_list+=[mdef_dict(8,'fc',{'input_nums':360448//pow(4,tree_def_list[branch_node]['parent']),'num_class':10},-1,-1,7)]
    return tree_def_list


    
# if __name__=='__main__':
#     for i in range(1,4):
#         tree_def_list=sample_net_tree(1)
#         x=torch.rand(1,3,32,32)
#         net=Net_generator(tree_def_list,1)
#         result=net(x)
#         print(result)

def Sample_net(i):
    tree_def_list=sample_net_tree(i)
    return Net_generator(tree_def_list,i)