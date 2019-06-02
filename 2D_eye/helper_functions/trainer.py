from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import sys
import time

import numpy as np
import tqdm

import torch
from torch.autograd import Variable
import pickle
import multiprocessing as mp
from torch.multiprocessing import Process, Queue

from .torchsummary import summary
from .metrics import evaluate_segmentation
from .trainer_utils import adjust_learning_rate, Performance_Metrics, to_one_hot

import matplotlib.pyplot as plt
plt.switch_backend('agg')

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
sys.path.append("%s/../../.." % file_path)
import utils.photometric_transforms as ph_transforms
from models.SegNet import SegNet_Small

INPUT_CHANNELS = 3
NUM_CLASSES = 4

def train(
        train_loader,
        model,
        criterion,
        optimizer,
        args,
        epoch,
        prev_loss,
):
    '''
    Pytorch model training module
    :param train_loader: pytorch dataloader
    :param model: pytorch model
    :param criterion: loss function
    :param optimizer: NN training optimizer function
    :param args: input arguments from __main__
    :param epoch: epoch counter used to display training progress
    :param prev_loss: logs loss prior to invocation of train on batch
    '''
    model.train()
    loss_f = 0
    t_start = time.time()
    CUDA = args.gpu is not None

    for batch_idx, data in tqdm.tqdm(enumerate(train_loader), total = train_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        if CUDA:
            image = image.cuda(args.gpu)
            label = label.cuda(args.gpu)

        predicted_tensor, softmaxed_tensor = model(image)
        optimizer.zero_grad()
        loss = criterion(predicted_tensor, label)
        loss.backward()
        optimizer.step()
        loss_f += loss.item()

    delta = time.time() - t_start

    normalized_loss = loss_f * 1.0 / train_loader.__len__()

    is_better = normalized_loss < prev_loss
    if is_better:
        #print ('saving best train checkpoint...')
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model_train_best.pth"))

    print("Epoch #{}\t Train Loss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f* 1.0 / train_loader.__len__(), delta))
    return normalized_loss


def validate(
        val_loader,
        model,
        criterion,
        args,
        epoch,
        val_loss = float("inf")
):
    '''
    Pytorch model validation module
    :param val_loader: pytorch dataloader
    :param model: pytorch model
    :param criterion: loss function
    :param args: input arguments from __main__
    :param epoch: epoch counter used to display training progress
    :param val_loss: logs loss prior to invocation of train on batch
    '''
    model.eval()
    t_start = time.time()
    CUDA = args.gpu is not None
    loss_f = 0
    save_img_bool = True
    batch_idx_img = np.random.randint(0,val_loader.__len__())
    for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total=val_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        if CUDA:
            image = image.cuda(args.gpu)
            label = label.cuda(args.gpu)


        with torch.set_grad_enabled(False):
            predicted_tensor, softmaxed_tensor = model(image)

        loss = criterion(predicted_tensor, label)
        loss_f += loss.item()

        delta = time.time() - t_start

        try:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
        except TypeError as e:
            print (e)
            save_img_bool=False

        id_img = np.random.randint(0, args.batch_size)
        for idx, predicted_mask in enumerate(predicted_tensor):
            if idx == id_img and batch_idx == batch_idx_img:
                input_image, target_mask = image[idx], label[idx]
                c,h,w = input_image.size()
                if save_img_bool:
                    fig = plt.figure()
                    a = fig.add_subplot(1,3,1)
                    if c == 1:
                        plt.imshow(input_image.detach().cpu().transpose(1,2).transpose(0, 2)[:,:,0],cmap='gray')
                    else:
                        plt.imshow(input_image.detach().cpu().transpose(1,2).transpose(0, 2),cmap='gray')
                    a.set_title('Input Image')

                    a = fig.add_subplot(1,3,2)
                    predicted_mx = predicted_mask.detach().cpu().numpy()
                    predicted_mx = predicted_mx.argmax(axis=0)
                    plt.imshow(predicted_mx)
                    a.set_title('Predicted Mask')

                    a = fig.add_subplot(1,3,3)
                    target_mx = target_mask.detach().cpu().numpy()
                    plt.imshow(target_mx)
                    a.set_title('Ground Truth')
                    fig.savefig(os.path.join(args.output_dir, "prediction_{}_{}_{}.png".format(epoch+1, batch_idx, idx)))
                    plt.close(fig)

    print("Epoch #{}\t Val Loss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f* 1.0 / val_loader.__len__(), delta))

    new_val_loss=loss_f* 1.0 / val_loader.__len__()


    if new_val_loss<val_loss:
        print(val_loss, ',', new_val_loss)
        print('saving checkpoint ....')
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model_val_best.pth"))

    return new_val_loss

def run_training(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        args,
):
    '''
    Pytorch model run training session
    :param train_loader: pytorch dataloader
    :param val_loader: pytorch dataloader
    :param model: pytorch model
    :param criterion: loss function
    :param optimizer: SGD optimizer function
    :param args: input arguments from __main__
    '''
    prev_loss = float("inf")
    val_loss_f = float("inf")
    train_loss = []
    val_loss = []


    for epoch in range(args.num_epochs):
        if train_loader is not None:
            adjust_learning_rate(optimizer, epoch, args)
            loss_f = train(
                train_loader,
                model,
                criterion,
                optimizer,
                args,
                epoch,
                prev_loss,
            )
            train_loss.append(loss_f)
            prev_loss = np.array(train_loss).min()
        ## Validate per epoch
        if epoch%10 == 0:
            if val_loader is not None:
                val_loss_f = validate(
                    val_loader,
                    model,
                    criterion,args,
                    epoch,
                    val_loss_f,
                )
                val_loss.append(val_loss_f)
                val_loss_f = np.array(val_loss).min()

    ## run validation on last epoch
    val_loss_f = validate(
        val_loader,
        model,
        criterion,
        args,
        epoch,
        val_loss_f
    )
    val_loss.append(val_loss_f)

    return train_loss,val_loss

def run_testing(
        val_loader,
        model,
        args,
        get_images = False,
):
    '''
    Module to run testing on trained pytorch model
    current implementation is not memory efficient..
    do not expect testing code to run on significantly larger dataset at present...
    :param val_loader: dataloader
    :param model: pretrained pytorch model
    :param: args: input arguments from __main__
    :param get_images: Boolean, True implies model generated segmentation mask results will be dumped out
    '''
    model.eval()
    CUDA = args.gpu is not None
    # preds = []
    # acts = []
    # inp_data = []

    ## Compute Metrics:
    Global_Accuracy=[]; Class_Accuracy=[]; Precision=[]; Recall=[]; F1=[]; IOU=[]
    pm = Performance_Metrics(
        Global_Accuracy,
        Class_Accuracy,
        Precision,
        Recall,
        F1,
        IOU,
    )

    for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total = val_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        if CUDA:
            image = image.cuda(args.gpu)
            label = label.cuda(args.gpu)

        with torch.set_grad_enabled(False):
            predicted_tensor, softmaxed_tensor = model(image)

        image = image.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        pred = np.argmax(predicted_tensor.detach().cpu().numpy(), axis=1)

        if batch_idx == 0:
            collate_image = image.copy()
            collate_preds = pred.copy()
            collate_labels = label.copy()
        else:
            collate_image = np.vstack([collate_image, image.copy()])
            collate_preds = np.vstack([collate_preds, pred.copy()])
            collate_labels = np.vstack([collate_labels, label.copy()])

        for idx in np.arange(pred.shape[0]):
            ga, ca, prec, rec, f1, iou = evaluate_segmentation(
                pred[idx, :],
                label[idx, :],
                NUM_CLASSES,
            )
            pm.GA.append(ga)
            pm.CA.append(ca)
            pm.Precision.append(prec)
            pm.Recall.append(rec)
            pm.F1.append(f1)
            pm.IOU.append(iou)

    if get_images:
        return pm, collate_image, collate_preds, collate_labels
    else:
        return pm

def run_prediction(
        val_loader,
        model,
        args,
):
    '''
    Module to generate segmentation results on user provided data
    Quite similar to run_testing but no metrics generated since true ground truth not available
    current implementation is not memory efficient..
    do not expect testing code to run on significantly larger dataset at present...
    The code by default dumps resut images
    :param val_loader: dataloader
    :param model: pretrained pytorch model
    :param: args: input arguments from __main__
    '''
    model.eval()
    CUDA = args.gpu is not None
    preds = []
    inp_image = []

    for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total = val_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        if CUDA:
            image = image.cuda(args.gpu)
            label = label.cuda(args.gpu)

        with torch.set_grad_enabled(False):
            predicted_tensor, softmaxed_tensor = model(image)

        inp_image.append(image.detach().cpu().numpy())
        preds.append(predicted_tensor.detach().cpu().numpy())

    for i in range(len(preds)):
        if i == 0:
            collate_inp_image = inp_image[i]
            collate_preds = preds[i]
        else:
            collate_inp_image = np.vstack([collate_inp_image, inp_image[i]])
            collate_preds = np.vstack([collate_preds, preds[i]])
    collate_preds = np.argmax(collate_preds, 1)

    return collate_inp_image, collate_preds

def run_finetune(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        args,
):
    '''
    Fine tune model
    :param train_loader: pytorch dataloader
    :param val_loader: pytorch dataloader
    :param model: pytorch model
    :param criterion: loss function
    :param optimizer: SGD optimizer function
    :param args: input arguments from __main__
    '''
    prev_loss = float("inf")
    val_loss_f = float("inf")
    train_loss = []
    val_loss = []

    # # freeze layers but only leave certain layer to learn
    # for name, param in model.named_modules():
    #     if name != 'conv11d':
    #         param.requires_grad = False

    for epoch in range(args.num_epochs):
        if train_loader is not None:
            adjust_learning_rate(optimizer, epoch, args)
            loss_f = train(
                train_loader,
                model,
                criterion,
                optimizer,
                args,
                epoch,
                prev_loss,
            )
            train_loss.append(loss_f)
            prev_loss = np.array(train_loss).min()
        ## Validate per epoch
        if epoch%10 == 0:
            if val_loader is not None:
                val_loss_f = validate(
                    val_loader,
                    model,
                    criterion,args,
                    epoch,
                    val_loss_f,
                )
                val_loss.append(val_loss_f)
                val_loss_f = np.array(val_loss).min()

    ## run validation on last epoch
    val_loss_f = validate(
        val_loader,
        model,
        criterion,
        args,
        epoch,
        val_loss_f
    )
    val_loss.append(val_loss_f)

    return train_loss,val_loss
