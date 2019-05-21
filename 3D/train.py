from __future__ import print_function

import os
import sys

import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

# from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d

from dataset.shapeNetData import ShapeNetDataset, ShapeNetGTDataset

from models.pointnet import PointNetSeg
from models.discriminator import ConvDiscNet

import timeit
start = timeit.default_timer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = 'PointNetSeg'
BATCH_SIZE = 16
ITER_SIZE = 1
NUM_WORKERS = 6

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
NUM_INST_CLASSES = 16
NUM_SEG_CLASSES = 50
NUM_STEPS = 175088 # 20000
POWER = 0.9
# RANDOM_SEED = 1234

SAVE_PRED_EVERY = 1750 # every 2 epoches, batch size 16
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.01

PARTIAL_DATA=0.25

SEMI_START=8754 #10 epochs
LAMBDA_SEMI=0.1
MASK_T=0.1

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0 # 0
D_REMAIN=True

parser = argparse.ArgumentParser(description="PointNet-Semi Network")
parser.add_argument("--model", type=str, default=MODEL,
                    help="available options : PointNet")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                    help="Accumulate gradients for ITER_SIZE iterations.")
parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                    help="number of workers for multithread dataloading.")
parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--partial-id", type=str, default=None,
                    help="restore partial id list")
parser.add_argument("--is-training", action="store_true",
                    help="Whether to updates the running means and variances during the training.")
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                    help="Base learning rate for discriminator.")
parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                    help="lambda_adv for adversarial training.")
parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                    help="lambda_semi for adversarial training.")
parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                    help="lambda_semi for adversarial training.")
parser.add_argument("--mask-T", type=float, default=MASK_T,
                    help="mask T for semi adversarial training.")
parser.add_argument("--semi-start", type=int, default=SEMI_START,
                    help="start semi learning after # iterations")
parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                    help="start semi learning after # iterations")
parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                    help="Whether to train D with unlabeled data")
parser.add_argument("--momentum", type=float, default=MOMENTUM,
                    help="Momentum component of the optimiser.")
parser.add_argument("--not-restore-last", action="store_true",
                    help="Whether to not restore last (FC) layers.")
parser.add_argument("--num-instance-classes", type=int, default=NUM_INST_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-seg-classes", type=int, default=NUM_SEG_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                    help="Number of training steps.")
parser.add_argument("--power", type=float, default=POWER,
                    help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-mirror", action="store_true",
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true",
                    help="Whether to randomly scale the inputs during the training.")
# parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
#                     help="Random seed to have reproducible results.")
parser.add_argument("--restore-from-D", type=str, default=None,
                    help="Where restore model parameters from.")
parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                    help="Save summaries and checkpoint every often.")
parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                    help="Where to save snapshots of the model.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                    help="Regularisation parameter for L2-loss.")
parser.add_argument("--gpu", type=int, default=0,
                    help="choose gpu device.")
parser.add_argument("--num-pts", type=int, dest="num_pts", default=2048,
                    help="#points in each instance")
parser.add_argument("--noise", action="store_true", dest="noise", default=False,
                   help="add noise in data augmentation")
parser.add_argument("--rotate", action="store_true", dest="rotate", default=False,
                   help="rotate shape in data augmentation")

opts = parser.parse_args()
print(opts)

results_folder = os.path.join(BASE_DIR, "results")
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)
writer_folder = os.path.join(results_folder, "logs")
if not os.path.isdir(writer_folder):
    os.mkdir(writer_folder)
writer = SummaryWriter(writer_folder)


class CrossEntropyMask(nn.Module):

    def __init__(self, size_average=True, ignore_label=999):
        super(CrossEntropyMask, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(b, c, npts)
                target:(b, npts)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 3
        assert target.dim() == 2
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))

        b, c, npts = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask] # (bxnpts)xc
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).contiguous()
        predict = predict[target_mask.view(b, npts, 1).repeat(1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def loss_calc(pred, label, device, mask=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    pred = pred.transpose(2,1)
    label = Variable(label.long()).to(device)
    if not mask:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = CrossEntropyMask().to(device)

    return criterion(pred, label)


def loss_bce(pred, label, device):
    """
    :return: BCE loss
    """
    label = Variable(label.float()).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(opts.learning_rate, i_iter, opts.num_steps, opts.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(opts.learning_rate_D, i_iter, opts.num_steps, opts.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label, num_classes):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], label.shape[1], num_classes), dtype=label.dtype)
    # for i in range(num_classes):
    #     one_hot[:,i,...] = (label==i)
    for b in range(label.shape[0]):
        for p in range(label.shape[1]):
            one_hot[b, p, label[b,p]] = 1
    #handle ignore labels
    # return torch.FloatTensor(one_hot)
    return torch.tensor(one_hot, dtype=torch.float)


def make_D_label(label, label_shape, device):
    # ignore_mask = np.expand_dims(label_shape, axis=1)
    D_label = np.ones(label_shape.shape)*label
    # D_label[ignore_mask] = 255
    # D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)
    D_label = Variable(torch.tensor(D_label, dtype=torch.float)).to(device)

    return D_label


def main():
    cudnn.enabled = True
    # create network
    model = PointNetSeg(NUM_SEG_CLASSES)

    writer.add_graph(model, (Variable(torch.Tensor(1,2048,3), requires_grad=True),
                              Variable(torch.Tensor(1,1,16), requires_grad=True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    cudnn.benchmark = True

    # init D
    model_D = ConvDiscNet(input_dim=NUM_SEG_CLASSES)
    writer.add_graph(model_D, Variable(torch.Tensor(1, 2048, 50), requires_grad=True))
    writer.close()

    if opts.restore_from_D is not None:
        model_D.load_state_dict(torch.load(opts.restore_from_D))
    model_D.to(device)
    model_D.train()


    if not os.path.exists(opts.snapshot_dir):
        os.makedirs(opts.snapshot_dir)

    train_dataset = ShapeNetDataset(num_classes=NUM_INST_CLASSES,
                                    npts=opts.num_pts,
                                    mode = "train",
                                    aug_noise = opts.noise,
                                    aug_rotate = opts.rotate)
    train_dataset_size = len(train_dataset)
    print("#Total train: {:6d}".format(train_dataset_size))

    train_gt_dataset = ShapeNetGTDataset(num_classes=NUM_INST_CLASSES,
                                         npts=opts.num_pts)

    if opts.partial_data is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=opts.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=opts.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        partial_size = int(opts.partial_data * train_dataset_size)

        if opts.partial_id is not None:
            train_ids = pickle.load(open(opts.partial_id))
            print('loading train ids from {}'.format(opts.partial_id))
        else:
            train_ids = list(range(train_dataset_size))
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(os.path.join(opts.snapshot_dir, 'train_id.pkl'), 'wb'))

        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                        batch_size=opts.batch_size, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=opts.batch_size, sampler=train_remain_sampler, num_workers=NUM_WORKERS, pin_memory=True)
        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=opts.batch_size, sampler=train_gt_sampler, num_workers=NUM_WORKERS, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)


    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.001, betas=(0.9,0.999))
    optimizer_D.zero_grad()

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    for i_iter in range(opts.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(opts.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # do semi first
            if (opts.lambda_semi > 0 or opts.lambda_semi_adv > 0 ) and i_iter >= opts.semi_start_adv :
                try:
                    batch = next(trainloader_remain_iter)
                except:
                    trainloader_remain_iter = iter(trainloader_remain)
                    batch = next(trainloader_remain_iter)


                # only access to img
                points, cls, _ = batch
                points, cls = Variable(points).float(), Variable(cls).float()
                points, cls = points.to(device), cls.to(device)

                pred = model(points, cls) # BxNxC
                pred_remain = pred.detach()

                D_out = model_D(F.softmax(pred, dim=2))
                D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy() # BxN
                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool) # Bx2048

                loss_semi_adv = opts.lambda_semi_adv * loss_bce(D_out, make_D_label(gt_label, ignore_mask_remain, device), device)
                loss_semi_adv = loss_semi_adv/opts.iter_size

                # writer.add_scalar('loss_semi_adv', loss_semi_adv.item(), i_iter)

                loss_semi_adv_value += loss_semi_adv.item() / opts.lambda_semi_adv

                if opts.lambda_semi <= 0 or i_iter < opts.semi_start:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    semi_ignore_mask = (D_out_sigmoid < opts.mask_T)

                    semi_gt = pred.data.cpu().numpy().argmax(axis=2)
                    semi_gt[semi_ignore_mask] = 999

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = torch.FloatTensor(semi_gt)

                        loss_semi = opts.lambda_semi * loss_calc(pred, semi_gt, device, mask=True)
                        loss_semi = loss_semi/opts.iter_size
                        # writer.add_scalar('loss_semi', loss_semi.item(), i_iter)
                        loss_semi_value += loss_semi.item() / opts.lambda_semi
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source

            try:
                # _, batch = trainloader_iter.next()
                batch = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)


            points, cls_gt, seg_gt = batch
            points, cls_gt, seg_gt = Variable(points).float(), Variable(cls_gt).float(), Variable(seg_gt).type(torch.LongTensor)
            points, cls_gt, seg_gt = points.to(device), cls_gt.to(device), seg_gt.to(device)
            # ignore_mask = (seg_gt.numpy() == 255)
            ignore_mask = np.zeros(seg_gt.shape).astype(np.bool)
            pred = model(points, cls_gt) # [5,21,41,41] -> [5, 21, 321, 321]

            loss_seg = loss_calc(pred, seg_gt, device, mask=False)
            # writer.add_scalar('loss_seg', loss_seg.item(), i_iter)

            D_out = model_D(F.softmax(pred, dim=2)) # [5, 1, 10, 10] -> [5, 1, 321, 321]

            loss_adv_pred = loss_bce(D_out, make_D_label(gt_label, ignore_mask, device), device)
            # writer.add_scalar('loss_adv_pred', loss_adv_pred.item(), i_iter)

            loss = loss_seg + opts.lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss/opts.iter_size
            loss.backward()
            loss_seg_value += loss_seg.item() / opts.iter_size
            loss_adv_pred_value += loss_adv_pred.item() / opts.iter_size


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()

            if opts.D_remain:
                try:
                    pred_remain
                    pred = torch.cat((pred, pred_remain), 0)
                    ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)
                except NameError:
                    print("No pred_remain during training D")

            D_out = model_D(F.softmax(pred, dim=2))
            loss_D = loss_bce(D_out, make_D_label(pred_label, ignore_mask, device), device)
            # loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/opts.iter_size/2
            loss_D_pred = loss_D.item()
            # writer.add_scalar('loss_D_pred', loss_D_pred, i_iter)
            loss_D.backward()
            loss_D_value += loss_D.item()


            # train with gt
            # get gt labels
            try:
                batch = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = iter(trainloader_gt)
                batch = next(trainloader_gt_iter)

            _, cls_gt, seg_gt = batch
            D_gt_v = Variable(one_hot(seg_gt, NUM_SEG_CLASSES)).float().to(device)
            ignore_mask_gt = np.zeros(seg_gt.shape).astype(np.bool)
            # ignore_mask_gt = (seg_gt.numpy() == 255)

            D_out = model_D(D_gt_v)
            loss_D = loss_bce(D_out, make_D_label(gt_label, ignore_mask_gt, device), device)
            loss_D = loss_D/opts.iter_size/2
            # writer.add_scalar('loss_D_gt', loss_D.item(), i_iter)
            # writer.add_scalar('loss_D_total', loss_D.item()+loss_D_pred, i_iter)
            loss_D.backward()
            loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(opts.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, \
                loss_seg = {2:.3f}, \
                loss_adv_p = {3:.3f}, \
                loss_D = {4:.3f}, \
                loss_semi = {5:.3f}, \
                loss_semi_adv = {6:.3f}'.format(
                    i_iter, opts.num_steps,
                    loss_seg_value,
                    loss_adv_pred_value,
                    loss_D_value,
                    loss_semi_value,
                    loss_semi_adv_value))

        if i_iter >= opts.num_steps-1:
            print('save model ...')
            torch.save(model.state_dict(),os.path.join(opts.snapshot_dir, 'ShapeNet_'+str(opts.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(opts.snapshot_dir, 'ShapeNet_'+str(opts.num_steps)+'_D.pth'))
            break

        if i_iter % opts.save_pred_every == 0 and i_iter!=0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),os.path.join(opts.snapshot_dir, 'ShapeNet_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(opts.snapshot_dir, 'ShapeNet_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')
    # writer.close()

if __name__ == '__main__':
    main()
