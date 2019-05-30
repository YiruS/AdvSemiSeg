from __future__ import print_function

import os
import sys

import numpy as np
import glob
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)
import utils.photometric_transforms as ph_transforms

import torch


def one_hot(label, num_classes):
    one_hot = np.zeros((1, num_classes), dtype = label.dtype)
    one_hot[0, label] = 1
    return one_hot

class OpenEDSDataset(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    """

    def __init__(self,
                 root,
                 photometric_transform = None):
        self.root = root
        self.photometric_transform = photometric_transform

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.npy")
        self.label_list = glob.glob(os.path.join(self.root, "masks") + "/*.npy")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        im = np.load(self.img_list[index])
        label = np.load(self.label_list[index])

        if self.photometric_transform:
            im = self.photometric_transform(im)

        return im, label


class CalipsoDataset(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    """

    def __init__(self,
                 root,
                 photometric_transform = None):
        self.root = root
        self.photometric_transform = photometric_transform

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.npy")
        self.label_list = glob.glob(os.path.join(self.root, "labels") + "/*.npy")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        im = np.load(self.img_list[index])
        label = np.load(self.label_list[index])

        if self.photometric_transform is not None:
            im_t = self.photometric_transform(im)
            return im, im_t, label

        return im, label

if __name__ == '__main__':
    my_transforms = [
        ph_transforms.ChangeBrightness(2.0),
        ph_transforms.ToTensor(),
    ]

    openeds = OpenEDSDataset(
        root = "/home/yirus/Data/OpenEDS_SS_TL/train/"
    )
    print("#train: {}".format(len(openeds)))
    calipso = CalipsoDataset(
        root = "/home/yirus/Data/Calipso_TL",
        photometric_transform = ph_transforms.Compose(my_transforms),
    )
    print("#train: {}".format(len(calipso)))

    openeds_loader = torch.utils.data.DataLoader(openeds, batch_size=1)
    calipso_loader = torch.utils.data.DataLoader(calipso, batch_size=1)

    fig = plt.figure()
    for i, data in enumerate(openeds_loader):
        im, label = data
        im, label = np.asarray(np.squeeze(im.numpy(), axis = 0)), \
                    np.asarray(np.squeeze(label.numpy(), axis = 0))

        ax = plt.subplot(1, 2, 1)
        ax.imshow(im, cmap = "gray")
        ax = plt.subplot(1, 2, 2)
        ax.imshow(label)
        plt.pause(1)

        if i == 10:
            break
    plt.title("OpenEDS")
    plt.show()

    fig = plt.figure()
    for i, data in enumerate(calipso_loader):
        im, im_t, label = data
        im, im_t, label = np.asarray(np.squeeze(im.numpy(), axis=0)), \
                    np.asarray(np.squeeze(im_t.numpy(), axis=0)), \
                    np.asarray(np.squeeze(label.numpy(), axis=0))

        ax = plt.subplot(1, 3, 1)
        ax.imshow(im, cmap = "gray")
        ax = plt.subplot(1, 3, 2)
        ax.imshow(im_t, cmap = "gray")
        ax = plt.subplot(1, 3, 3)
        ax.imshow(label)
        plt.pause(1)

        if i == 10:
            break
    plt.title("Calipso")
    plt.show()
