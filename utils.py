import generator
#计算每层输出之后的input_size
def parameters_and_flops_calculator(mdef):
    input_size=32
    middle_size=0
    if mdef['type_name']=='conv':
        parameters_amount= pow(mdef['kernel_size'],2)*mdef['in_ch']*mdef['out_ch']
        flops_amount=parameters_amount*pow(input_size,2)
    
    if mdef['type_name']=='resblock':
        kernel_size=3
        strides=[mdef['stride']]+[1]*(mdef['num_blocks']-1)
        in_channels=mdef['in_ch']
        out_channels=mdef['out_ch']
        padding=1
        for i,stride in strides:
            if i==0:    
                parameters_count=pow(kernel_size,2)*in_channels*out_channels+kernel_size*out_channels*out_channels
                flops_amount+=parameters_count*pow(input_size,2)
                middle_size=(input_size-kernel_size+2*padding)//stride+1
                parameters_amount+=parameters_count
            else:
                parameters_count=pow(kernel_size,2)*out_channels*out_channels
                flops_amount+=parameters_count*pow(middle_size,2)
                middle_size=(input_size-kernel_size+2*padding)//stride+1
                parameters_amount+=parameters_count
            if stride!=1 or in_channels!=out_channels:
                parameters_count+=in_channels*out_channels
                flops_amount+=parameters_count*pow(middle_size,2)
                middle_size=(input_size-kernel_size+2*padding)//stride+1
                parameters_amount+=parameters_count

    
    if mdef['type_name']=='inceptiona':
        input_size=32
        padding=1
        padding_branch5x5=2
        in_channels=mdef['in_ch']
        pool_feature=mdef['po_ft']
        out_channels_branch1x1=64
        out_channels_branch5x5_1=48
        out_channels_branch5x5_2=64
        kernel_size_branch5x5=5
        kernel_size_branch3x3=3
        out_channels_branch3x3dbl_1=64
        out_channels_branch3x3dbl_2=96
        out_channels_branch3x3dbl_3=96
        #1x1卷积分支
        parameters_count_branch1x1=in_channels*out_channels_branch1x1
        flops_amount+=pow(input_size,2)*parameters_count_branch1x1

        #5x5卷积
        parameters_count_branch5x5_1=in_channels*out_channels_branch5x5_1
        flops_amount+=pow(input_size,2)*parameters_count_branch5x5_1
        parameters_count_branch5x5_2=in_channels*pow(kernel_size_branch5x5,2)*out_channels_branch5x5_2
        middle_size=(input_size-kernel_size_branch5x5+2*padding_branch5x5)//stride+1
        flops_amount+=pow(middle_size,2)*parameters_count_branch5x5_1
        parameters_count_branch5x5=parameters_count_branch5x5_1+parameters_count_branch5x5_2

        #3x3卷积
        parameters_count_branch3x3dbl_1=in_channels*out_channels_branch3x3dbl_1
        flops_amount+=pow(input_size,2)*parameters_count_branch3x3dbl_1
        middle_size=(input_size-kernel_size_branch5x5+2*padding_branch5x5)+1
        parameters_count_branch3x3dbl_2=out_channels_branch3x3dbl_1*out_channels_branch3x3dbl_2*pow(kernel_size_branch3x3,2)
        flops_amount+=pow(middle_size,2)*parameters_count_branch3x3dbl_2
        middle_size=(middle_size-kernel_size_branch3x3+2*padding)+1
        parameters_count_branch3x3dbl_3=out_channels_branch3x3dbl_2*out_channels_branch3x3dbl_3*pow(kernel_size_branch3x3,2)
        flops_amount+=pow(middle_size,2)*parameters_count_branch3x3dbl_3
        middle_size=(middle_size-kernel_size_branch3x3+2*padding)+1
        parameters_count_branch3x3=parameters_count_branch3x3dbl_1+parameters_count_branch3x3dbl_2+parameters_count_branch3x3dbl_3

        #pooling分支
        parameters_count_pooling=in_channels*pool_feature
        flops_amount+=pow(input_size,2)*parameters_count_pooling

        parameters_amount=parameters_count_branch1x1+parameters_count_branch3x3+parameters_count_branch5x5
    
    if mdef['type_name']=='inceptionb':
        in_channels=mdef['in_ch']
        kernel_size=3
        stride=2
        padding=1
        out_channels_branch3x3=384
        out_channels_branch3x3dbl_1=64
        out_channels_branch3x3dbl_2=96
        out_channels_branch3x3dbl_3=96
        #3x3卷积分支
        parameters_count_branch3x3=pow(kernel_size,2)*in_channels*out_channels_branch3x3
        flops_amount+=pow(input_size,2)*parameters_count_branch3x3

        #3x3dbl分支
        parameters_count_branch3x3dbl_1=in_channels*out_channels_branch3x3dbl_1
        flops_amount+=pow(input_size,2)*parameters_count_branch3x3dbl_1
        parameters_count_branch3x3dbl_2=out_channels_branch3x3dbl_1*out_channels_branch3x3dbl_2*pow(kernel_size_branch3x3,2)
        flops_amount+=pow(middle_size,2)*parameters_count_branch3x3dbl_2
        middle_size=(input_size-kernel_size+2*padding)//stride+1
        parameters_count_branch3x3dbl_3=out_channels_branch3x3dbl_2*out_channels_branch3x3dbl_3*pow(kernel_size_branch3x3,2)
        flops_amount+=pow(middle_size,2)*parameters_count_branch3x3dbl_3
        parameters_amount=parameters_count_branch3x3+parameters_count_branch3x3dbl_1+parameters_count_branch3x3dbl_2+parameters_count_branch3x3dbl_3
        
    
    if mdef['type_name']=='bottleneck':
        in_channels=mdef['in_ch']
        out_channels=mdef['out_ch']
        kernel_size=3
        padding=1
        parameters_count_1=in_channels*out_channels
        flops_amount+=pow(input_size,2)*parameters_count_1
        parameters_count_3x3=out_channels*out_channels*pow(kernel_size,2)
        flops_amount+=pow(middle_size,2)*parameters_count_branch3x3
        middle_size=(input_size-kernel_size+2*padding)//stride+1
        parameters_count_1x1_2=out_channels*out_channels
        flops_amount+=pow(middle_size,2)*parameters_count_1x1_2
        parameters_amount=parameters_count_1+parameters_count_3x3+parameters_count_1x1_2
    
    if mdef['type_name']=='fc':
        in_channels=mdef['in_ch']
        out_channels=mdef['out_ch']
        parameters_amount=in_channels*out_channels
        flops_amount=in_channels*out_channels
    
    return parameters_amount,flops_amount
    





# def sample_net_tree(branch_node):
#     tree_def_list=[]
#     tree_def_list+=[mdef_dict(0,'conv',{'kernel_size':3,'in_ch':3,'out_ch':64,'stride':1,'batch_normalize':True,'activation':'relu'},1,-1,-1)]  
#     tree_def_list+=[mdef_dict(1,'resblock',{'block':BasicBlock,'in_ch':64,'out_ch':64,'num_blocks':3,'stride':1},2,5,0)]    #360448  90112     1
#     tree_def_list+=[mdef_dict(2,'resblock',{'block':BasicBlock,'in_ch':64,'out_ch':128,'num_blocks':3,'stride':2},3,-1,1)]  #90112   正确      1/4
#     tree_def_list+=[mdef_dict(3,'resblock',{'block':BasicBlock,'in_ch':128,'out_ch':256,'num_blocks':3,'stride':2},4,-1,2)] #22528   90112     1/16
#     tree_def_list+=[mdef_dict(4,'resblock',{'block':BasicBlock,'in_ch':256,'out_ch':512,'num_blocks':3,'stride':2},8,-1,3)]
#     tree_def_list+=[mdef_dict(5,'fc',{'input_nums':8192,'num_class':10},-1,-1,4)]
#     tree_def_list+=[mdef_dict(6,'inceptiona',{'in_ch':tree_def_list[branch_node]['describe']['out_ch'],'po_ft':64},6,-1,branch_node)]
#     tree_def_list+=[mdef_dict(7,'inceptiona',{'in_ch':288,'po_ft':128},7,-1,6)]
#     #tree_def_list+=[mdef_dict(7,'bottleneck',{'in_ch':352,'out_ch':64},8,-1,2)]
#     print(tree_def_list[branch_node]['parent'])
#     tree_def_list+=[mdef_dict(8,'fc',{'input_nums':360448//pow(4,tree_def_list[branch_node]['parent']),'num_class':10},-1,-1,7)]
#     return tree_def_list
