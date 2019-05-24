import os
import sys

import numpy as np
import torch
from torch.autograd import Variable

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from dataset.shapeNetData import ShapeNetDataset, ShapeNetGTDataset
from models.pointnet import PointNetSeg
from models.discriminator import ConvDiscNet


def fastprint(str):
  print(str)
  sys.stdout.flush()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter, learning_rate_D, num_steps, power):
    lr = lr_poly(learning_rate_D, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def make_D_label(label, label_shape, device):
    D_label = np.ones(label_shape.shape)*label
    D_label = Variable(torch.tensor(D_label, dtype=torch.float)).to(device)
    return D_label

def one_hot(label, num_classes):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], label.shape[1], num_classes), dtype=label.dtype)
    for b in range(label.shape[0]):
        for p in range(label.shape[1]):
            one_hot[b, p, label[b,p]] = 1
    return torch.tensor(one_hot, dtype=torch.float)

def create_dataset(num_inst_classes, num_pts, mode, is_noise = False, is_rotate = False):
    dataset = ShapeNetDataset(
        num_classes = num_inst_classes,
        num_pts = num_pts,
        mode = mode,
        aug_noise = is_noise,
        aug_rotate = is_rotate
    )
    return dataset

def create_GT_dataset(num_inst_classes, num_pts):
    dataset = ShapeNetGTDataset(
        num_classes = num_inst_classes,
        num_pts = num_pts
    )
    return dataset

def create_dataloader(dataset, batch_size, num_workers, shuffle = True, pin_memory = True, sampler = None):
    if sampler is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            pin_memory = pin_memory,
            sampler = sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            pin_memory = pin_memory
        )
    return dataloader

def create_model(type, num_seg_classes):
    if type == "generator":
        model = PointNetSeg(num_seg_classes)
    elif type == "discriminator":
        model = ConvDiscNet(input_dim=num_seg_classes)
    else:
        raise ValueError("Invalid model type {}!".format(type))
    return model