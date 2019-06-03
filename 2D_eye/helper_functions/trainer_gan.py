import os
import sys

import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt

from helper_functions.trainer_utils import make_D_label
from .metrics import evaluate_segmentation
from .trainer_utils import adjust_learning_rate, Performance_Metrics

import torch
from torch.autograd import Variable
import torch.nn.functional as F

GT_LABEL = 1
PRED_LABEL = 0
NUM_CLASSES = 4

def train_SS(
        model_SS,
        model_D,
        trainloader_D1,
        trainloader_D2,
        optimizer_SS,
        criterion_CE_D1,
        criterion_CE_D2,
        criterion_ADV,
        lambda_ce,
        lambda_adv,
        device,
        epoch,
        prev_loss,
        args,
):
    """
    Train SS network, including loss_ce and loss_adv with for dataset 2.
    :param model_SS: model of SS
    :param model_D: model of Discriminator
    :param trainloader_D1: dataloader of D1, where |D1| > |D2|
    :param trainloader_D2: dataloader of D2
    :param optimizer_SS, optimizer of SS network
    :param criterion_CE: loss function for loss_ce
    :param criterion_ADV: loss function for loss_adv (D2 only)
    :param lambda_ce: coefficient for loss_ce of D2
    :param lambda_adv: coefficient for loss_adv of D2
    :param device: CPU / GPU
    :return: loss_ce for D1, loss_ce for D2, loss_adv for D2
    """
    loss_ce_value_1 = []
    loss_ce_value_2 = []
    loss_adv_value = []

    model_SS.train()
    model_D.train()

    model_SS.to(device)
    model_D.to(device)

    for i, combine_batch in tqdm.tqdm(enumerate(itertools.zip_longest(trainloader_D1, trainloader_D2)), total = trainloader_D1.__len__()):
        batch_d1, batch_d2 = combine_batch

        for param in model_SS.parameters():
            param.requires_grad = True
        for param in model_D.parameters():
            param.requires_grad = False

        optimizer_SS.zero_grad()

        if batch_d1 is None:
            raise ValueError("|D1| should > |D2|!")
        else: # only compute loss_ce for |D1|
            image, label = batch_d1
            image, label = Variable(image).float(), \
                           Variable(label).type(torch.LongTensor)
            image, label = image.to(device), \
                           label.to(device)
            predicted_tensor, softmaxed_tensor = model_SS(image)

            loss_ce_1 = criterion_CE_D1(predicted_tensor, label)
            loss_SS = loss_ce_1
            loss_ce_value_1.append(loss_ce_1.item())
        if batch_d2 is not None:
            image, label = batch_d2
            image, label = Variable(image).float(), \
                           Variable(label).type(torch.LongTensor)
            image, label = image.to(device), \
                           label.to(device)
            predicted_tensor, softmaxed_tensor = model_SS(image)
            # loss_ce
            loss_ce_2 = criterion_CE_D2(predicted_tensor, label)
            # loss_adv
            D_out = model_D(F.softmax(predicted_tensor, dim=2))
            gt_label = make_D_label(GT_LABEL, D_out, device).float()
            loss_adv = criterion_ADV(D_out, gt_label)

            loss_SS += lambda_ce*loss_ce_2 + lambda_adv*loss_adv

            loss_ce_value_2.append(loss_ce_2.item())
            loss_adv_value.append(loss_adv.item())

        # try:
        #     loss_ce_2
        #     loss_SS = loss_ce_1 + lambda_ce*loss_ce_2 + lambda_adv*loss_adv
        # except:
        #     loss_SS = loss_ce_1

        loss_SS.backward()
        optimizer_SS.step()

    loss_ce_avg_1 = np.average(loss_ce_value_1)
    loss_ce_avg_2 = np.average(loss_ce_value_2)
    loss_adv_avg = np.average(loss_adv_value)
    loss_SS_avg = loss_ce_avg_1+loss_ce_avg_2+loss_adv_avg

    is_better = loss_SS_avg < prev_loss
    if is_better:
        # print ('saving best train checkpoint...')
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model_SS.state_dict(), os.path.join(args.save_dir, "modelSS_train_best_epoch_{}.pth").format(epoch))

    print("Epoch #{}\t SSNet Overall Loss: {:.4f}\t CE1: {:.4f} \t CE2: {:.4f} \t ADV: {:.4f}".format(
        epoch + 1,
        loss_SS_avg,
        loss_ce_avg_1,
        loss_ce_avg_2,
        loss_adv_avg)
    )

    return loss_ce_avg_1, loss_ce_avg_2, loss_adv_avg

def train_discriminator(
        model_SS,
        model_D,
        trainloader_D1,
        trainloader_D2,
        optimizer_D,
        criterion,
        device,
        epoch,
        prev_loss,
        args,
):
    """
    Train discriminator
    :return: L_D (trained w/ D1 and D2)
    """
    loss_D_value_gt = []
    loss_D_value_pred = []
    loss_D_value_total = []

    model_SS.train()
    model_D.train()

    model_SS.to(device)
    model_D.to(device)

    for i, combine_batch in tqdm.tqdm(enumerate(itertools.zip_longest(trainloader_D1, trainloader_D2)), total = trainloader_D1.__len__()):
        batch_d1, batch_d2 = combine_batch

        for param in model_SS.parameters():
            param.requires_grad = False
        for param in model_D.parameters():
            param.requires_grad = True

        optimizer_D.zero_grad()

        if batch_d1 is None:
            raise ValueError("|D1| should > |D2|!")
        elif batch_d1 is not None: # |D1| treated as GT
            image, label = batch_d1
            image, label = Variable(image).float(), \
                           Variable(label).type(torch.LongTensor)
            image, label = image.to(device), \
                           label.to(device)
            pred, softmaxed_tensor = model_SS(image)
            pred = pred.detach()

            D_out = model_D(F.softmax(pred, dim=2))
            gt_label = make_D_label(GT_LABEL, D_out, device).float()

            loss_D_GT = criterion(D_out, gt_label)
            loss_D = loss_D_GT
            loss_D_value_gt.append(loss_D_GT.item())
        if batch_d2 is not None:  # |D2| treated as PRED
            image, label = batch_d2
            image, label = Variable(image).float(), \
                           Variable(label).type(torch.LongTensor)
            image, label = image.to(device), \
                           label.to(device)
            pred, softmaxed_tensor = model_SS(image)
            pred = pred.detach()

            D_out = model_D(F.softmax(pred, dim=2))
            pred_label = make_D_label(PRED_LABEL, D_out, device).float()

            loss_D_PRED = criterion(D_out, pred_label)
            loss_D_value_pred.append(loss_D_PRED.item())
            loss_D += loss_D_PRED


        loss_D.backward()
        optimizer_D.step()
        loss_D_value_total.append(loss_D.item())

    loss_D_gt_avg, loss_D_pred_avg, loss_D_all_avg = \
        np.average(loss_D_value_gt), np.average(loss_D_value_pred), np.average(loss_D_value_total)

    is_better = loss_D_all_avg < prev_loss
    if is_better:
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model_D.state_dict(), os.path.join(args.save_dir, "modelD_train_best_epoch_{}.pth").format(epoch))

    print("Epoch #{}\t DNet Overall Loss: {:.4f}\t GT loss: {:.4f} \t Pred loss: {:.4f}".format(
        epoch + 1,
        loss_D_all_avg,
        loss_D_gt_avg,
        loss_D_pred_avg)
    )
    return loss_D_gt_avg, loss_D_pred_avg, loss_D_all_avg

def validate(
        val_loader,
        model_SS,
        criterion,
        device,
        args,
        epoch,
        val_loss = float("inf")
):
    '''
    Pytorch model validation module
    :param val_loader: pytorch dataloader for Calipso (D2)
    :param model: pytorch model
    :param criterion: loss function (CE)
    :param args: input arguments from __main__
    :param epoch: epoch counter used to display training progress
    :param val_loss: logs loss prior to invocation of train on batch
    '''
    model_SS.eval()
    model_SS.to(device)

    loss_f = 0
    save_img_bool = True

    batch_idx_img = np.random.randint(0, val_loader.__len__())
    for batch_idx, data in enumerate(val_loader):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(device), \
                       label.to(device)

        with torch.set_grad_enabled(False):
            predicted_tensor, softmaxed_tensor = model_SS(image)

        loss = criterion(predicted_tensor, label)
        loss_f += loss.item()

        try:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
        except TypeError as e:
            print (e)
            save_img_bool = False

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

    print("Epoch #{}\t Val Loss: {:.8f}".format(
        epoch + 1,
        loss_f* 1.0 / val_loader.__len__())
    )

    new_val_loss=loss_f* 1.0 / val_loader.__len__()

    if new_val_loss<val_loss:
        print(val_loss, ',', new_val_loss)
        print('saving checkpoint ....')
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(model_SS.state_dict(), os.path.join(args.save_dir, "modelSS_val_best.pth"))

    return new_val_loss

def run_training_gan(
        trainloader_D1,
        trainloader_D2,
        valloader_D2,
        model_SS,
        model_D,
        criterion_CE_D1,
        criterion_CE_D2,
        criterion_ADV,
        criterion_D,
        optimizer_SS,
        optimizer_D,
        device,
        args,
):
    '''
    Pytorch model run training session
    :param trainloader_D1: pytorch dataloader for D1 (OpenEDS)
    :param trainloader_D2: pytorch dataloader for D2 (Calipso)
    :param valloader_D2: pytorch dataloader for D2 (only validate D2)
    :param model_SS: pytorch model for SS
    :param model_D: pytorch model for Discriminator
    :param criterion_CE_D1: loss function (CE) for D1
    :param criterion_CE_D2: loss function (CE) for D2
    :param criterion_ADV: loss function for adversarial loss (train SS)
    :param criterion_D: loss function for Discriminator
    :param optimizer_SS: optimizer function for SS
    :param optimizer_D: optimizer function for Discriminator
    :param device: CPU or GPU
    :param args: input arguments from __main__
    '''
    prev_loss_ss = float("inf")
    prev_loss_discriminator = float("inf")

    train_loss_ss_all = []
    train_loss_ce = []
    train_loss_adv = []
    train_loss_d_all = []
    train_loss_d_gt = []
    train_loss_d_pred = []

    val_loss_f = float("inf")
    val_loss = []

    model_SS.train()
    model_D.train()

    model_SS.to(device)
    model_D.to(device)

    for epoch in range(args.num_epochs):
        if trainloader_D1 is not None and trainloader_D2 is not None:
            # adjust_learning_rate(optimizer, epoch, args)
            loss_ce_D1, loss_ce_D2, loss_adv = train_SS(
                model_SS = model_SS,
                model_D = model_D,
                trainloader_D1 = trainloader_D1,
                trainloader_D2 = trainloader_D2,
                optimizer_SS = optimizer_SS,
                criterion_CE_D1 = criterion_CE_D1,
                criterion_CE_D2 = criterion_CE_D2,
                criterion_ADV = criterion_ADV,
                lambda_ce = args.lambda_ce,
                lambda_adv = args.lambda_adv,
                device = device,
                epoch = epoch,
                prev_loss = prev_loss_ss,
                args = args,
            )
            train_loss_ss_all.append(loss_ce_D1+loss_ce_D2+loss_adv)
            train_loss_ce.append(loss_ce_D1+loss_ce_D2)
            train_loss_adv.append(loss_adv)
            prev_loss_ss = np.min(np.array(train_loss_ss_all))

            loss_D_gt, loss_D_pred, loss_D_total = train_discriminator(
                model_SS = model_SS,
                model_D = model_D,
                trainloader_D1 = trainloader_D1,
                trainloader_D2 = trainloader_D2,
                optimizer_D = optimizer_D,
                criterion = criterion_D,
                device = device,
                epoch = epoch,
                prev_loss = prev_loss_discriminator,
                args = args,
            )
            train_loss_d_all.append(loss_D_total)
            train_loss_d_gt.append(loss_D_gt)
            train_loss_d_pred.append(loss_D_pred)
            prev_loss_discriminator = np.min(np.array(train_loss_d_all))

        ## Validate per epoch
        if epoch % 10 == 0:
            if valloader_D2 is not None:
                val_loss_f = validate(
                    val_loader = valloader_D2,
                    model_SS = model_SS,
                    criterion = criterion_CE_D2,
                    device = device,
                    epoch = epoch,
                    args = args,
                    val_loss = val_loss_f,
                )
                val_loss.append(val_loss_f)
                val_loss_f = np.min(np.array(val_loss))

    ## run validation on last epoch
    val_loss_f = validate(
        val_loader = valloader_D2,
        model_SS = model_SS,
        criterion = criterion_CE_D2,
        device = device,
        epoch = epoch,
        args = args,
        val_loss = val_loss_f,
    )
    val_loss.append(val_loss_f)

    return train_loss_ss_all, train_loss_d_all, val_loss

def run_testing_gan(
        valloader_D2,
        model_SS,
        device,
        args,
        get_images = False,
):
    '''
    Module to run testing on trained pytorch model for Calipso (D2)
    :param train_loader: pytorch dataloader
    :param val_loader: pytorch dataloader
    :param model: pytorch model
    :param criterion: loss function
    :param optimizer: SGD optimizer function
    :param args: input arguments from __main__
    '''
    model_SS.eval()
    model_SS.to(device)

    Global_Accuracy = []
    Class_Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    IOU = []
    pm = Performance_Metrics(
        Global_Accuracy,
        Class_Accuracy,
        Precision,
        Recall,
        F1,
        IOU,
    )

    for batch_idx, data in enumerate(valloader_D2):
        image, label = data
        image, label = Variable(image).float(), \
                       Variable(label).type(torch.LongTensor)

        image, label = image.to(device), \
                       label.to(device)

        with torch.set_grad_enabled(False):
            predicted_tensor, softmaxed_tensor = model_SS(image)

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