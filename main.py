'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import time

import torchvision
import torchvision.transforms as transforms

import copy 

import os
import argparse

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--Rate', default=0.5, type=float, help='noisy rate')
parser.add_argument('--Iter', default=40, type=int, help='Iteration Number')
parser.add_argument('--index', default=0, type=int, help='index')
parser.add_argument('--Num', default=5000, type=int, help='Num')

args = parser.parse_args()

Rate=args.Rate
Iter=args.Iter
Num=args.Num
# BATCH_DELAY=50.
exp_weight=1-100/Num
GAP=10
SetIndex=args.index

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
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

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

trainset0=torch.load("./data/CIFAR_train_Noisy_%d_%d_All.pth"%(100*Rate,SetIndex))
# trainset1=torch.load("./data/MNIST_train_Noisy_%d_%d_Train.pth"%(100*Rate,SetIndex))
# valiset=torch.load("./data/MNIST_train_Noisy_%d_%d_Vali.pth"%(100*Rate,SetIndex))

trainloader0 = torch.utils.data.DataLoader(
    trainset0, batch_size=100, shuffle=True, num_workers=2)
# trainloader1 = torch.utils.data.DataLoader(
#     trainset1, batch_size=100, shuffle=True, num_workers=2)
# valiloader = torch.utils.data.DataLoader(
#     valiset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net1 =ResNet18()
net1 = net1.to(device)
if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    cudnn.benchmark = True

# net1_echo=copy.deepcopy(net1)
    
criterion = nn.CrossEntropyLoss()
# optimizer1 = optim.SGD(net1.parameters(), lr=args.lr,
#                       momentum=0.9)
# optimizer2 = optim.SGD(net2.parameters(), lr=args.lr,
#                       momentum=0.9)

optimizer1 = optim.Adam(net1.parameters(), lr=args.lr)

FLAG_SCH=False
# scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=100*Iter)
# scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100*Iter)
# scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda epoch: 1/(1+epoch/100))
# scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda epoch: 1/(1+epoch/100))
# scheduler1_echo = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1_echo, T_max=Iter)

record_batch=[]
record_real=[]

acc_real=0

def test0(epoch):
    global acc_real
    net1.eval()
    test_loss_0 = 0
    correct_real = 0
    correct_real_ens = 0
    test_loss_rel = 0
    correct_vali = 0
    vali_loss = 0
    
    with torch.no_grad():
        # real testing set
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, targets)
            test_loss_0 += loss1.item()
            _, predicted1 = outputs1.max(1)

            total += targets.size(0)            
            correct_real += predicted1.eq(targets).sum().item()
    
    acc_real=100.*correct_real/total
    record_real.append(100.*correct_real/total)

        
best_acc=0
acc_dist_ave=0
batch_count=0

timeStart=time.time()

def Save(): 
    name="TrainingRecord_Swap%d_R%d_B%d_%d.pth"%(Num,100*Rate,batch_count+1,SetIndex)
    torch.save([record_batch,record_real],name)

for epoch in range(start_epoch, start_epoch+int(Iter)):
    timeNow=time.time()
    print('\nEpoch: %d, Time: %d s' % (epoch,timeNow-timeStart))
    net1.train() # 启用 BatchNormalization 和 Dropout
    train_loss = 0
    correct = 0
    correct_dist = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader0):
        batch_count+=1
        if batch_count%5000==4999:
            Save()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer1.zero_grad()
        outputs1 = net1(inputs)
        loss1 = criterion(outputs1, targets)
        loss1.backward()
        optimizer1.step()      

        train_loss += loss1.item()
        _, predicted1 = outputs1.max(1)
        total += targets.size(0)
        correct += predicted1.eq(targets).sum().item()
        
        if batch_idx%GAP==GAP-1:
            record_batch.append(batch_count)
            test0(epoch)
            
            timeNow=time.time()
            print("Batch: %d, tr_acc: %.3f, acc: %.3f%%" %(batch_count,train_loss/(batch_idx+1),acc_real),"\r",end="")
            
            correct = 0
            correct_dist = 0
            total = 0
            
            if FLAG_SCH:
                scheduler1.step()
                scheduler2.step()
Save()