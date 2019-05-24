import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

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

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
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