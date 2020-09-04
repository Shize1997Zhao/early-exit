import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from mainnetwork import *
from generator import *
import utils

import os
import argparse



parser=argparse.ArgumentParser(description='Early Exit CIFAR10 Training')
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--depth',default=4,type=int,help='depth of BinarySearchTree')
parser.add_argument('--accuracy',default=0.98,type=float,help='target accuracy')
parser.add_argument('--entropy',default=0.2,type=float,help='threshold_entropy')
args=parser.parse_args()

device='cuda' if torch.cuda.is_available() else 'cpu'
best_acc1=0
best_acc2=0
start_epoch=0
exitnum=[]
threshold_entropy=0.2

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



# net=Sample_net(1)
# net=net.to(device)
# if device=='cuda':
#     net=torch.nn.DataParallel(net)
#     cudnn.benchmark=True

# if args.resume:
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs[0], targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs[0].max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = []
#     correct = []
#     total = []
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs[0], targets)

#             test_loss += loss.item()
#             _, predicted = outputs[0].max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
        
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc

# for epoch in range(start_epoch,start_epoch+200):
#     train(epoch)
#     test(epoch)

for i in range(1,4):
    net=Sample_net(i)
    for mdef in net.tree_def_list:
        parameters=[]
        flops=[]
        parameters_count,flops_count=utils.parameters_and_flops_calculator(mdef)
        parameters_amount+=parameters_count
        flops_amount+=flops_count
        if mdef['type_name']=='fc':
            parameters.append(parameters_amount)
            flops.append(flops_amount)
            parameters_amount=0
            flops_amount=0
    #net = resnet20_cifar()
    net=net.to(device)
    if device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True
    #print(next(net.parameters()).is_cuda)#False
    
    if args.resume:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        print('train')
        net.train()
        train_loss = 0
        #train_loss2 = 0
        correct1 = 0
        correct2 = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #loss=criterion(outputs[0], targets)+criterion(outputs[1], targets)
            #loss.backward()
            #optimizer.step()
            #loss2.backward()
            #optimizer.step()
            outputs1=outputs[0]
            outputs2=outputs[1]
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            if loss2<threshold_entropy:
                exitnum+=batch_idx
                exitrate=100.*len(exitnum)/batch_idx
            loss =exitrate(loss1 + loss2)            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #train_loss2 += loss2.item()
            _, predicted1 = outputs[0].max(1)
            _, predicted2 = outputs[1].max(1)
            total += targets.size(0)
            correct1 += predicted1.eq(targets).sum().item()
            correct2 += predicted2.eq(targets).sum().item()
            

    def test(epoch):
        print('test')
        global best_acc1
        global best_acc2
        net.eval()
        test_loss = 0
        #test_loss1 = []
        correct1 = 0
        correct2 = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                outputs1=outputs[0]
                outputs2=outputs[1]
                loss = criterion(outputs1, targets) + criterion(outputs2, targets)

                test_loss += loss.item()
                #test_loss2 += loss2.item()
                _, predicted1 = outputs[0].max(1)
                _, predicted2 = outputs[1].max(1)
                total += targets.size(0)
                correct1 += predicted1.eq(targets).sum().item()
                correct2 += predicted2.eq(targets).sum().item()

        
        acc1 = 100.*correct1/total
        acc2 = 100.*correct2/total
        if acc1 > best_acc1:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc1,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc1 = acc1
        if acc2 > best_acc2:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc2,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc2 = acc2

    for epoch in range(start_epoch,start_epoch+200):
        train(epoch)
        test(epoch)
    

