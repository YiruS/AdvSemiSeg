from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2
import pickle
import sys
from PIL import Image
import numpy as np

def visualize_heatmap(img, heatmaps, heatmap_type='dot'):
    # ===============================================================
    # img: numpy array of size H x W x 3
    # heatmaps: numpy array of size n_part x H x W
    # ===============================================================
    # if heatmap is just segmentation label, then convert from shape H x W to 1 x H x W
    if len(heatmaps.shape) == 2:
        heatmaps = np.expand_dims(heatmaps, axis=0)

    # convert float 0-1 to grayscale
    if img.max() < 1.05:
        img = (img * 255).astype('uint8')
    heatmap_overlay = 0.5 * img
    result = img.copy()
    # visualize result
    for idx, part_map in enumerate(heatmaps):
        # part_map_resized = cv2.resize(
        #     part_map, (img.shape[1], img.shape[0]),
        #     interpolation=cv2.INTER_LINEAR
        # )
        part_map_resized=np.array(Image.fromarray(part_map.astype('uint8'))\
        .resize((img.shape[1],img.shape[0]),Image.BILINEAR))
        print (part_map_resized.max(),part_map_resized.min())
        part_map_color = color_heatmap(part_map_resized)
        heatmap_overlay += 0.5 * part_map_color
        if heatmap_type == 'dot':
            prediction = np.unravel_index(
                part_map_resized.argmax(), part_map_resized.shape
            )
            cv2.circle(
                result, (int(prediction[1]), int(prediction[0])), 1,
                (0, 255, 0), -1
            )
        elif heatmap_type == 'curve':
            # first find the start and end point of this band
            idx = np.where((part_map_resized > 0.2).sum(axis=0) > 5)[0]
            if len(idx) > 10:
                w_start, w_end = idx.min(), idx.max()
                h_new = []
                for w in range(w_start, w_end, 1):
                    h_new.append(part_map_resized[:, w].argmax())
                h_new = np.array(h_new)
                w_new = np.arange(w_start, w_end, 1)
                func = np.poly1d(np.polyfit(w_new, h_new, 3))
                h_updated = np.array([func(w) for w in w_new])
                for (w, h) in zip(w_new, h_updated):
                    cv2.circle(result, (int(w), int(h)), 1, (255, 0, 0), -1)
        elif heatmap_type == 'seg':
            pred_label = heatmaps.argmax(axis=0)
            result = color_heatmap(pred_label)
            print ('pickled results')
            o=open('/home/stalathi/nfs/Temp/data_dump.pkl','wb')
            pickle.dump([heatmap_overlay,result,pred_label],o)
            o.close()
            sys.exit(0)
    #return heatmap_overlay, result
    return part_map_resized,result


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def plot_curve(result_dir, data):
    # first row: train_losses, 2nd row: train_angle
    # 3rd: test_losses, 4th: test_angle
    train_losses, train_angles, test_losses, test_angles = \
        np.split(data, [1, 2, 3], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
    plt.plot(np.arange(len(test_losses)), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    ax.set_yscale('log')
    loss_fname = os.path.join(result_dir, 'loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(np.arange(len(train_losses)), train_angles, label='Train')
    plt.plot(np.arange(len(test_losses)), test_angles, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Angle')
    plt.legend()
    err_fname = os.path.join(result_dir, 'error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))


def draw_gaussian(img, pos, sigma):
    # draw gaussian mask around pos on img
    x_l, x_u = max(0, pos[0] - 3 * sigma), min(pos[0] + 3 * sigma, img.shape[1])
    y_l, y_u = max(0, pos[1] - 3 * sigma), min(pos[1] + 3 * sigma, img.shape[0])

    y, x = np.mgrid[y_l:y_u:1, x_l:x_u:1]
    print(y_l, y_u, x_l, x_u)
    img[y_l:y_u, x_l:x_u] = np.exp(
        -((x - pos[0])**2 + (y - pos[1])**2) / (2.0 * sigma**2)
    )
    return img


def color_heatmap(heatmap):
    cscale = 255 / (heatmap.max() - heatmap.min())
    gray_img = np.zeros(heatmap.shape, dtype='uint8')
    cv2.convertScaleAbs(heatmap, gray_img, cscale, -heatmap.min() / 255)
    out = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    return out
