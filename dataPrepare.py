'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as Data

import numpy

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--Rate', default=0.2, type=float, help='noisy rate')
parser.add_argument('--index', default=0, type=int, help='index')
parser.add_argument('--Num', default=50000, type=int, help='Num')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
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

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

Sample=[]
Label=[]
count=0

Rate=args.Rate
reRate=Rate*10/9.

SetIndex=args.index

NUM_total=len(trainset)
# NUM_total=args.Num
# NUM_vali=1000
# NUM_train=NUM_total-NUM_vali

for i in range(NUM_total):
    sample=trainset[i][0]
    r=numpy.random.rand()
    if r>reRate:
        tag=trainset[i][1]
    else:
        index=numpy.random.randint(0,10)
        tag=index
    if not tag==trainset[i][1]:
        count+=1
    Sample.append(sample)
    Label.append(tag)
    if i%5000==0:
        print(i,tag)
    
print("Error count: ",count)
trainSample0 = torch.stack(Sample)
trainLabel0 = torch.Tensor(Label).to(torch.int64)
train_dataset0 = Data.TensorDataset(trainSample0,trainLabel0)
# trainData1 = torch.stack(Sample[:NUM_train-1])
# trainLabel1 = torch.Tensor(Label[:NUM_train-1]).to(torch.int64)
# train_dataset1 = Data.TensorDataset(trainData1,trainLabel1)
# valiData1 = torch.stack(Sample[NUM_train:])
# valiLabel1 = torch.Tensor(Label[NUM_train:]).to(torch.int64)
# vali_dataset1 = Data.TensorDataset(valiData1,valiLabel1)

torch.save(train_dataset0,"./data/CIFAR_train_Noisy_%d_%d_All.pth" %(Rate*100,SetIndex))
# torch.save(train_dataset1,"./data/MNIST_train_Noisy_%d_%d_Train.pth" %(Rate*100,SetIndex))
# torch.save(vali_dataset1,"./data/MNIST_train_Noisy_%d_%d_Vali.pth" %(Rate*100,SetIndex))
    
print("Done.")