from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import time
import numpy as np
import copy

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_global_accuracy(flat_pred, flat_label):
    '''
    compute global prediction accuracy for feature segmentation
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return fraction of pixels that have same prediction
    '''
    try:
        assert len(flat_pred)==len(flat_label)
    except AssertionError as e:
        print (e)
        return
    total = len(flat_label)
    return float(sum(flat_pred==flat_label))/float(total)

def compute_class_accuracies(flat_pred, flat_label, num_classes):
    '''
    compute per class prediction accuracy for feature segmentation
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return fraction of pixels per class that have same prediction
    '''
    total = []
    for val in range(num_classes):
        total.append((flat_label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(flat_label)):
        if flat_pred[i] == flat_label[i]:
            count[int(flat_pred[i])] = count[int(flat_pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(flat_pred, flat_label):
    '''
    compute mean intersection over union (IOU) over all classes
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return: mean IOU
    '''
    unique_labels = np.unique(flat_label)
    num_unique_labels = len(unique_labels)

    Intersect = np.zeros(num_unique_labels)
    Union = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = flat_pred == val
        label_i = flat_label == val

        Intersect[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        Union[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(Intersect / Union)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    '''
    Compute various performance metrics
    :param pred: prediction matrix
    :param label: label matrix
    :num_classes: number of classes
    :score_averaging: type of weighting to compute average
    :returns performance metrics for quality of feature segmentation present in prediction
    '''
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou
