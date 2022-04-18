#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 20:46:36 2022

@author: root
"""

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt 
from cls_dataset import CLS_Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
parser = argparse.ArgumentParser(description=' Example')
parser.add_argument('--batchSize', type=int, default=64, metavar='N',  # 改一次送的图片张数
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N')  # 整个数据集迭代多少次
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.002')  # 学习速率,往小改
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')  # 正则化
parser.add_argument('--threads', type=int, default=20, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print(args)
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
set_random_seed(args.seed)          
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")
print("===> load datasets")


data_transform = transforms.Compose([
#    transforms.Resize((188, 128)),
#    transforms.RandomHorizontalFlip(),
#    transforms.Grayscale(num_output_channels=1),
    
    transforms.ToTensor()
])
    
data_transform1 = transforms.Compose([
#    transforms.Resize((188, 128)),
#    transforms.Grayscale(num_output_channels=1),
   
    transforms.ToTensor()
])
        

        
print("====>load traindatset ")
root = 'separated_data/karyogram/label.txt'  
train_dataset = CLS_Dataset(txt=root, transform=data_transform,mode='Train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,shuffle=True,drop_last=False,num_workers=args.threads)

print("====>load testdatset ")
test_dataset = CLS_Dataset(txt=root, transform=data_transform1,mode='Test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,shuffle=False,drop_last=False,num_workers=args.threads)


print("====>load valdatset ")
val_dataset =CLS_Dataset(txt=root, transform=data_transform1,mode='Val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,
                                           shuffle=False,drop_last=False,num_workers=args.threads)
from inception_Resnet import Inception_ResNetv2
from resnet18 import resnet18

model_names=['inception_resnetV2']
#resnet18(num_classes=4)
models=[Inception_ResNetv2(in_channels=1, classes=23)]
#models=[resnet18(num_classes=23)]
criterion = nn.CrossEntropyLoss()


print('===> Begining Training')
def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = 1
  k = args.epochs // 10
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.lr * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    

def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0.0
    train_loss = 0.0
    total =0
    train_iterator = tqdm(train_loader,leave=True, total=len(train_loader),position=0,ncols=10,miniters=10)
    iterators=0
    for  inputs, labels in tqdm(train_iterator):
        torch.cuda.empty_cache()
        iterators=iterators+1
    
      
        images=inputs.to(device)
        print(images.size)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
       
        train_loss += loss.item()
        total += labels.size(0)
        outputs=F.softmax(outputs)
        _, prediction = torch.max(outputs.data, 1)

        train_acc += torch.sum(prediction == labels)
        status="===> Epoch[{}]({}/{}): train_loss:{:.4f},mean_loss:{:.4f}, train_acc:{:.4f}".format(
        epoch, iterators, len(train_loader), loss.item(),train_loss/total,train_acc/total)
        print(status)
#        train_iterator.set_description(status)
        
        loss.backward()
        optimizer.step()

      
    scheduler.step()
    return train_loss/total

def test(epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        model.eval()
        test_acc = 0.0
        total =0
       
      
        for iteration, (images, labels) in enumerate(test_loader):
          
            images=images.to(device)
           
            labels=labels.to(device)
    
            outputs = model(images)
            outputs=F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
       
            print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f}".format(
                epoch, test_acc/total))
            
    return test_acc/total


def val(epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
       
        model.eval()
        val_acc = 0.0
        
       
        total =0
        
        for iteration, (images, labels) in enumerate(val_loader):
            images=images.to(device)
           
            labels=labels.to(device)
    
            outputs = model(images)
            outputs=F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            val_acc += torch.sum(prediction == labels)
         
            print("===> Epoch[{}] =====>Mean_val_Acc:{:.4f}".format(
                epoch, val_acc/total))
           
    return val_acc/total

def checkpoint(name):
    model_out_path = name
    torch.save(model.state_dict(), model_out_path)
    print("\n===>Checkpoint saved to {}".format(model_out_path))
          
if __name__ == '__main__' :
    for i in range(len(model_names)):
        model=models[i]
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        mode='cosineAnnWarm'
        if mode=='cosineAnn':
            scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        elif mode=='cosineAnnWarm':
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=3)
        import os
        test_best_acc=0
        log_dir='checkpoints/{}/model.pth'.format(model_names[i])
        last_log='checkpoints/{}/seq_last.pth'.format(model_names[i])
        best_log='checkpoints/{}/seq_best.pth'.format(model_names[i])
       
        test_acc_list=[]
        val_acc_list=[]
        
        
        if os.path.exists(log_dir):

            model.load_state_dict(torch.load(log_dir))
            if 0:
                for param in model.parameters(): 
                    param.requires_grad = False
                print(model.linear)
#            num_feature = model.linear.in_features  #获取fc层的输入个数
#            model.linear = nn.Linear(num_feature, 23)  #重新定义fc层
            print('load_weight')
            model.to(device)  
            print('finetuing model')
        count=0
        for epoch in range(1, args.epochs + 1): 
            
            total_loss=train(epoch)
            test_acc=test(epoch)
            test_acc_list.append(test_acc.cpu().data.numpy())
            
            val_acc=val(epoch)
            
    
           
            if epoch==1:
             
                 test_best_acc=test_acc
                 val_best_acc=  val_acc
            
            if val_acc>val_best_acc:
               checkpoint(log_dir)
            
               test_best_acc= test_acc
               val_best_acc= val_acc
             
            else:
              
                test_best_acc= test_best_acc
                val_best_acc= val_best_acc
                count=count+1
                
            checkpoint(last_log)
            if count>1000:
                checkpoint(best_log)
                break

            print("\n===>val_acc not be improved  to {:.4f}".format( val_best_acc))
            print("\n===>test_acc not be improved  to {:.4f}".format( test_best_acc))
           
            
