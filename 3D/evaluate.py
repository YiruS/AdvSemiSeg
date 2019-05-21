import argparse
import numpy as np
import sys
import os
from packaging import version
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils import data, model_zoo

from models.pointnet import PointNetSeg
from dataset.shapeNetData import ShapeNetDataset, ShapeNetGTDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = 'PointNetSeg'
# IGNORE_LABEL = 999
NUM_CLASSES = 50
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = "VOC_134750.pth"
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'

NUM_INST_CLASSES = 16
NUM_SEG_CLASSES = 50


object_names = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp',
                'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
part_colors = [[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]]

cls_shape_cnt = {}
for shape_cls in object_names:
    cls_shape_cnt[shape_cls] = 0

seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

shape_ious = {cat:[] for cat in seg_classes.keys()}

parser = argparse.ArgumentParser(description="PointNet-Semi Network for Evaluation")
parser.add_argument("--model", type=str, default=MODEL,
                    help="Model of Generator")
# parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
#                     help="The index of the label to ignore during the training.")
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-pts", type=int, dest="num_pts", default=2048,
                    help="#points in each instance")
parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                    help="Where restore model parameters from.")
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                    help="Directory to store results")
parser.add_argument("--gpu", type=int, default=0,
                    help="choose gpu device.")

opts = parser.parse_args()
print(opts)


def visualize_prediction(pts, gt, pred, cls_gt, cnt, iou):
    visualization_folder = os.path.join(BASE_DIR, "visualization")
    if not os.path.isdir(visualization_folder):
        os.mkdir(visualization_folder)
    visualization_gt_folder = os.path.join(visualization_folder, "gt")
    if not os.path.isdir(visualization_gt_folder):
        os.mkdir(visualization_gt_folder)
    visualization_pred_folder = os.path.join(visualization_folder, "pred")
    if not os.path.isdir(visualization_pred_folder):
        os.mkdir(visualization_pred_folder)


    from dataset.shapeNetData import pts2img, part_colors
    pnt_colormap_gt = []
    for j in range(gt.shape[0]):
        pnt_colormap_gt.append(part_colors[gt[j]])
    cur_obj = object_names[cls_gt]
    title = "gt_" + cur_obj + "_cnt_" + str(cnt) + ".png"
    im = pts2img(pts, pnt_colormap_gt)
    im.save(os.path.join(visualization_gt_folder, title))
    print("Object ({:s})".format(cur_obj))

    pnt_colormap_pred = []
    for j in range(pred.shape[0]):
        pnt_colormap_pred.append(part_colors[pred[j]])
    title = "pred_" + cur_obj + "_cnt_" + str(cnt) + "_iou_" + "{:.3f}".format(iou) + ".png"
    im = pts2img(pts, pnt_colormap_pred)
    im.save(os.path.join(visualization_pred_folder, title))
    print("Object ({:s})".format(cur_obj))


def get_iou(gt, pred, cls_gt):
    cls_name = object_names[cls_gt]
    shape_cat = seg_classes[cls_name]
    part_ious = [0.0 for _ in range(len(shape_cat))]
    for l in shape_cat:
        if (np.sum(pred == l) == 0) and (np.sum(gt == l) == 0):  # part is not present, no prediction as well
            part_ious[l - shape_cat[0]] = 1.0
        else:
            part_ious[l - shape_cat[0]] = np.sum((gt == l) & (pred == l)) / float(
                np.sum((gt == l) | (pred == l)))
    shape_ious[cls_name].append(np.mean(part_ious))
    return np.mean(part_ious)

def main():
    """Create the model and start the evaluation process."""
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)


    model = PointNetSeg(NUM_SEG_CLASSES)
    snapshots_path = os.path.join(BASE_DIR, "snapshots")
    saved_state_dict = torch.load(os.path.join(snapshots_path, opts.restore_from))
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    test_dataset = ShapeNetDataset(num_classes=NUM_INST_CLASSES,
                                    npts=opts.num_pts,
                                    mode = "test",
                                    aug_noise = False,
                                    aug_rotate = False)
    testloader = data.DataLoader(test_dataset,
                                    batch_size=1, shuffle=False, pin_memory=True)



    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('processd {}'.format(index))
        points, cls_gt, seg_gt = batch
        with torch.no_grad():
            points, cls_gt, seg_gt = Variable(points).float(), Variable(cls_gt).float(), Variable(seg_gt).type(torch.LongTensor)
            points, cls_gt, seg_gt = points.to(device), cls_gt.to(device), seg_gt.to(device)
            seg_pred = model(points, cls_gt)
            points, gt, pred, cls_gt = points.data.cpu().numpy(), seg_gt.data.cpu().numpy(), np.argmax(seg_pred.cpu().numpy(),axis=2), cls_gt.data.cpu().numpy()
            points, gt, pred, cls_gt = np.squeeze(points), np.squeeze(gt), np.squeeze(pred), np.where(np.squeeze(cls_gt)==1)[0][0]
            # print("shape: {}".format(object_names[cls_gt]))
            cls_shape_cnt[object_names[cls_gt]] += 1
            iou = get_iou(gt, pred, cls_gt)

            visualize_prediction(points, gt, pred, cls_gt, cls_shape_cnt[object_names[cls_gt]], iou)

    print("===== Shape Statistics =====")
    for obj, cnt in cls_shape_cnt.items():
        print("#{} = {}".format(obj, cnt))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print(len(all_shape_ious))
    mean_shape_ious = np.mean(np.array(list(shape_ious.values())))
    for cat in sorted(shape_ious.keys()):
        print('eval mIoU of %s:\t %f' % (cat, shape_ious[cat]))
    print('eval mean mIoU: %f' % (mean_shape_ious))
    print('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))


if __name__ == '__main__':
    main()
