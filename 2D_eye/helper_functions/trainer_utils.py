from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import sys
import time
import numpy as np
import torch
import pickle
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue

from .metrics import evaluate_segmentation
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
sys.path.append("%s/../../.." % file_path)
from utils.generic_utils import id_generator


NUM_CLASSES = 4

class Parallel_Engine(object):
    '''
    Parallel Engine class wrapper to compute ML model accuracy metrics
    '''
    def __init__(self,
                 preds,
                 acts,
                 num_classes,
                 pm
                 ):
        self.preds=preds
        self.acts=acts
        self.num_classes=num_classes
        self.pm=pm
    def __call__(self,i):
        ga, ca, prec, recall, f1 ,iou = evaluate_segmentation(
            self.preds[i],
            self.acts[i],
            self.num_classes
        )
        self.pm.GA.append(ga)
        self.pm.CA.append(ca)
        self.pm.GA.append(ga)
        self.pm.Precision.append(ca)
        self.pm.Recall.append(recall)
        self.pm.F1.append(f1)
        self.pm.IOU.append(iou)
        return None

class Performance_Metrics(object):
    '''
    Structure class to compile various accuracy metrics for evaluation
    Feature_Segmentation ML model
    '''
    def __init__(self, ga,ca, prec,recall, f1, iou ):
        self.GA=ga
        self.CA=ca
        self.Precision=prec
        self.Recall=recall
        self.F1=f1
        self.IOU=iou

def adjust_learning_rate(optimizer, epoch, args):
    """
    Module that sets the learning rate to the initial LR decayed by 10 every 'interval' epochs
    """
    interval=3 if args.num_epochs//3==0 else args.num_epochs//3
    lr = args.lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def define_criterion(train_set,args):
    '''
    Module to define the loss function criterion for training ML models
    Currently support Cross_Entropy loss and SoftDice loss
    '''
    criterion=None
    class_weight=None
    try:
        args.gpu
        args.loss
    except AttributeError as e:
        print (e)
        sys.exit(0)

    if args.CUDA:
        class_weight = 1.0 / train_set.get_class_probability().cuda(args.gpu)
        if 'CE' in args.loss.upper():
            criterion = torch.nn.CrossEntropyLoss(
                weight=class_weight
            ).cuda(args.gpu)
        elif 'SD' in args.loss.upper():
            criterion = SoftDiceLoss(
                num_classes=NUM_CLASSES,
                weight=class_weight,
                cuda=args.CUDA,
                cuda_device=args.gpu)
        else:
            print ('Undefined loss... exiting')
            sys.exit(0)
    else:
        class_weight = 1.0 / train_set.get_class_probability()
        if 'CE' in args.loss.upper():
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        elif 'SD' in args.loss.upper():
            criterion = SoftDiceLoss(num_classes=NUM_CLASSES,weight=class_weight)
        else:
            print ('Undefined loss.. exiting')
            sys.exit(0)
    return class_weight, criterion

def to_one_hot(
        y,
        num_classes=4,
        cuda=False,
        cuda_device=0
):
    '''
    Module that takes integer y (tensor or variable) with n dims and convert
    it to 1-hot representation with n+1 dims.
    '''
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = num_classes #n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    if cuda:
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1).cuda(cuda_device)
    else:
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot

class SoftDiceLoss(nn.Module):
    '''
    Pytorch compatible class object to define Dice loss
    More information: https://link.springer.com/chapter/10.1007/978-3-319-46976-8_19
    '''
    def __init__(self, num_classes=4,weight=None,smooth=1e-2, size_average=True, cuda=False,cuda_device=0):
        super(SoftDiceLoss, self).__init__()
        self.weight=weight
        self.smooth=smooth
        self.size_average=size_average
        self.num_classes=num_classes
        self.cuda=cuda
        self.cuda_device=cuda_device

    def forward(self,input,target):
        input=torch.softmax(input,1)
        (B,C,W,H)=input.shape

        if self.weight is not None:
            ref=self.weight[0]*input[:,0,:,:]
            ref=ref.reshape(B,1,W,H)
            for i in range(1,len(self.weight)):
                ref=torch.cat([ref,self.weight[i]*input[:,i,:,:].reshape(B,1,W,H)],dim=1)
        if len(target.size())==3:
            target=to_one_hot(target,self.num_classes,self.cuda,self.cuda_device).transpose(1,3).transpose(2,3)

        if self.weight is not None:
            iflat = ref.contiguous().view(-1)
        else:
            iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat*tflat).sum()
        return 1-((2* intersection + self.smooth)/(iflat.sum()+tflat.sum()+ self.smooth))
