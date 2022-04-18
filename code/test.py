#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from cls_dataset import CLS_Dataset
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser(description=' Example')
parser.add_argument('--batchSize', type=int, default=1, metavar='N',  # 改一次送的图片张数
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N')  # 整个数据集迭代多少次
parser.add_argument('--lr', type=float, default=0.00002, help='Learning Rate. Default=0.002')  # 学习速率,往小改
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

    transforms.ToTensor()
])

print("====>load testdatset ")
root = 'separated_data/karyogram/label.txt' 
print("====>load testdatset ")
test_dataset = CLS_Dataset(txt=root, transform=data_transform,mode='Test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,shuffle=False,drop_last=False,num_workers=args.threads)

def test():
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        test_acc = 0.0
        
       
        total =0
       
        tmp_prediction=[]
        tmp_scores=[]
        total_label=[]
        for iteration, (images, labels) in enumerate(test_loader):
            images=images.to(device)
           
            labels=labels.to(device)
    
            outputs = model(images)
            outputs=F.softmax(outputs)
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
  
            print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f}".format(
                iteration, test_acc/total))
            
            prediction= prediction.cpu().data.numpy()
            tmp_prediction.append(prediction)
            tmp_scores.append(outputs[0].data.cpu().numpy())
            total_label.append(labels[0].data.cpu().numpy())
            
           
    return test_acc/total,np.array(tmp_prediction),np.array(total_label),np.array(tmp_scores)



        
if __name__ == '__main__' :
        import os
        from sklearn.metrics import confusion_matrix
        import pandas as pd
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support
        from scipy import interp
        import matplotlib.pyplot as plt 
        from inception_Resnet import Inception_ResNetv2
        model_names=['inception_resnetV2']
        models=[Inception_ResNetv2(in_channels=1, classes=23)]
        criterion = nn.CrossEntropyLoss()
        

        for n in range(len(model_names)):
            model=models[n]
            model = model.to(device)
            log_dir='checkpoints/{}/model.pth'.format(model_names[n])
            last_log='checkpoints/{}/seq_last.pth'.format(model_names[n])
            best_log='checkpoints/{}/seq_best.pth'.format(model_names[n])
            model.load_state_dict(torch.load(log_dir,map_location={'cuda:0':'cuda:1'}))
            print('load_weight')
            model.to(device)  
          
            test_acc,total_pred,total_label, total_scores=test()
          
            print('test acc:{}'.format(test_acc))