from __future__ import print_function

import os
import sys
import random

import numpy as np
import glob
import time
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)
import utils.photometric_transforms as ph_transforms

NUM_CLASSES = 4

import torch

def dataloader(
        args,
        type = "train",
):
    '''
    helper module to load data using the EyeSegmentData generator
    :param args: List of args from __main__
    :param type: specify the type of dataloader
    '''
    if args.dataset == "Calipso" or args.dataset == "OpenEDS":
        if type.lower() == "train":
            datafile = "%s/train" % args.data_root
        elif type.lower() == "val":
            if os.path.isdir("%s/validation" % args.data_root):
                datafile = "%s/validation" % args.data_root
            else:
                datafile = "%s/test" % args.data_root
        elif type.lower() == "test":
            datafile = "%s/test" % args.data_root
        else:
            datafile = args.data_root
    elif args.dataset == "Fused":
        if type.lower() == "train":
            datafile_openeds = "%s/train" % args.openeds_root
            datafile_calipso = "%s/train" % args.calipso_root
        elif type.lower() == "val":
            if os.path.isdir("%s/validation" % args.openeds_root):
                datafile_openeds = "%s/validation" % args.openeds_root
            else:
                datafile_openeds = "%s/test" % args.openeds_root
            if os.path.isdir("%s/validation" % args.calipso_root):
                datafile_calipso = "%s/validation" % args.calipso_root
            else:
                datafile_calipso = "%s/test" % args.calipso_root
        elif type.lower() == "test":
            datafile_openeds = "%s/test" % args.openeds_root
            datafile_calipso = "%s/test" % args.calipso_root

    start_loader = time.time()

    if args.dataset == "OpenEDS":
        data_set = OpenEDSDataset(
            root = datafile,
            train_bool = True,
        )
    elif args.dataset == "Calipso":
        data_set = CalipsoDataset(
            root = datafile,
            photometric_transform = None, #ph_transforms.Compose(args.transforms),
            train_bool = True,
        )
    elif args.dataset == "Fused":
        data_set = fused_dataset(
            openeds_root = datafile_openeds,
            calipso_root = datafile_calipso,
            photometric_transform = None, #ph_transforms.Compose(args.transforms),
            train_bool=True,
        )

    print("Done loading %s data in %d sec" % (type.lower(), time.time() - start_loader))
    data_dataloader = torch.utils.data.DataLoader(
        data_set,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True,
    )
    return data_set, data_dataloader

def one_hot(label, num_classes):
    one_hot = np.zeros((1, num_classes), dtype = label.dtype)
    one_hot[0, label] = 1
    return one_hot

class OpenEDSDataset(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    Note:for segmentation, target is numpy array of size (th, tw) with pixel
    values representing the label index
    :param root: Essential, path to data_root
    :param ph_transform: list of transforms of photometric augmentation
    :param load_n: integer to load subset of images
    :param random_samples: Boolean, default False,
    :param train_bool: Boolean true if the data loader is for training dataset; to generate
    class prob and mean image
    """

    def __init__(self,
                 root,
                 photometric_transform = None,
                 load_n = None,
                 random_samples=False,
                 train_bool = False,
                 ):
        self.root = root
        self.photometric_transform = photometric_transform
        self.load_n = load_n
        self.random_samples = random_samples
        self.train_bool = train_bool

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.npy")
        self.label_list = glob.glob(os.path.join(self.root, "masks") + "/*.npy")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

        print("Found {} images and {} labels".format(len(self.img_list), len(self.label_list)))
        if self.train_bool:
            self.counts = self.__compute_class_probability()
            # self.mean_image = self.__compute_image_mean()

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        im = np.load(self.img_list[index])
        label = np.load(self.label_list[index])

        if self.photometric_transform:
            im = self.photometric_transform(im)

        return np.expand_dims(im, axis=0), label

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        if self.load_n is None:
            load_n = 10
        else:
            load_n = self.load_n
        if self.random_samples:
            sampleslist = np.random.randint(0, self.__len__(), load_n)
        else:
            sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i==0:
                img_mean=img
            else:
                img_mean+=img
        return 1.0*img_mean/self.__len__()


    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)


class CalipsoDataset(torch.utils.data.Dataset):
    """
    Calipso dataset
    """

    def __init__(self,
                 root,
                 photometric_transform = None,
                 load_n = None,
                 random_samples = False,
                 train_bool = False,
                 ):
        self.root = root
        self.photometric_transform = photometric_transform
        self.load_n = load_n
        self.random_samples = random_samples
        self.train_bool = train_bool

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.npy")
        self.label_list = glob.glob(os.path.join(self.root, "labels") + "/*.npy")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

        print("Found {} images and {} labels".format(len(self.img_list), len(self.label_list)))
        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        im = np.load(self.img_list[index])
        label = np.load(self.label_list[index])

        if self.photometric_transform is not None:
            im_t = self.photometric_transform(im)
            # return im, im_t, label
        else:
            im_t = im.copy()

        return np.expand_dims(im_t, axis=0), label

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        if self.load_n is None:
            load_n = 10
        else:
            load_n = self.load_n
        if self.random_samples:
            sampleslist = np.random.randint(0, self.__len__(), load_n)
        else:
            sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i==0:
                img_mean=img
            else:
                img_mean+=img
        return 1.0*img_mean/self.__len__()


    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)

class fused_dataset(torch.utils.data.Dataset):
    """
       Fuse Calipso_GT and OpenEDS together for baseline 3:
        train model on {Calipso_GT, OpenEDS}, test on Calipso only
    """

    def __init__(self,
                 openeds_root,
                 calipso_root,
                 photometric_transform=None,
                 load_n=None,
                 random_samples=False,
                 train_bool=False,
                 ):
        self.openeds_root = openeds_root
        self.calipso_root = calipso_root
        self.photometric_transform = photometric_transform
        self.load_n = load_n
        self.random_samples = random_samples
        self.train_bool = train_bool

        self.img_list_openeds = glob.glob(os.path.join(self.openeds_root, "images") + "/*.npy")
        self.label_list_openeds = glob.glob(os.path.join(self.openeds_root, "masks") + "/*.npy")

        self.img_list_calipso = glob.glob(os.path.join(self.calipso_root, "images") + "/*.npy")
        self.label_list_calipso = glob.glob(os.path.join(self.calipso_root, "labels") + "/*.npy")

        assert len(self.img_list_calipso) == len(self.label_list_calipso), \
            "Unmatched #images = {} with #labels = {} in Calipso!".format(
                len(self.img_list_calipso),
                len(self.label_list_calipso)
            )
        assert len(self.img_list_openeds) == len(self.label_list_openeds), \
            "Unmatched #images = {} with #labels = {} in OpenEDS!".format(
                len(self.img_list_openeds),
                len(self.label_list_openeds)
            )
        self.img_list, self.label_list = [], []
        for im in self.img_list_calipso: self.img_list.append(im)
        for im in self.img_list_openeds: self.img_list.append(im)
        for lb in self.label_list_calipso: self.label_list.append(lb)
        for lb in self.label_list_openeds: self.label_list.append(lb)

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {} in fused!".format(
                len(self.img_list),
                len(self.label_list)
            )

        self.__shuffle_paired_list()

        print("Found {} images and {} labels".format(len(self.img_list), len(self.label_list)))
        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = np.load(self.img_list[index]).astype(np.float)
        label = np.load(self.label_list[index]).astype(np.uint8)

        if self.photometric_transform is not None:
            im_t = self.photometric_transform(im)
        else:
            im_t = im.copy()

        return np.expand_dims(im_t, axis=0), label

    def __shuffle_paired_list(self):
        paired_list = list(zip(self.img_list, self.label_list))
        random.shuffle(paired_list)
        self.img_list, self.label_list = zip(*paired_list)

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        if self.load_n is None:
            load_n = 10
        else:
            load_n = self.load_n
        if self.random_samples:
            sampleslist = np.random.randint(0, self.__len__(), load_n)
        else:
            sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)

if __name__ == '__main__':
    my_transforms = [
        ph_transforms.ChangeBrightness(2.0),
        ph_transforms.ToTensor(),
    ]

    fused = fused_dataset(
        openeds_root = "/home/yirus/Datasets/OpenEDS_SS_TL/train",
        calipso_root = "/home/yirus/Datasets/Calipso/GT_0.25/train",
        photometric_transform=None,
        load_n=None,
        random_samples=False,
        train_bool=False,
    )
    fused_loader = torch.utils.data.DataLoader(fused, batch_size=1)
    fig = plt.figure()
    for i, data in enumerate(fused_loader):
        im, label = data
        im, label = np.asarray(np.squeeze(im.numpy())), \
                    np.asarray(np.squeeze(label.numpy(), axis=0))

        ax = plt.subplot(1, 2, 1)
        ax.imshow(im, cmap="gray")
        ax = plt.subplot(1, 2, 2)
        ax.imshow(label)
        plt.pause(1)

        if i == 100:
            break
    plt.title("OpenEDS")
    plt.show()


    # openeds = OpenEDSDataset(
    #     root = "/home/yirus/Data/OpenEDS_SS_TL/train/"
    # )
    # print("#train: {}".format(len(openeds)))
    # calipso = CalipsoDataset(
    #     root = "/home/yirus/Data/Calipso_TL",
    #     photometric_transform = ph_transforms.Compose(my_transforms),
    # )
    # print("#train: {}".format(len(calipso)))
    #
    # openeds_loader = torch.utils.data.DataLoader(openeds, batch_size=1)
    # calipso_loader = torch.utils.data.DataLoader(calipso, batch_size=1)
    #
    # fig = plt.figure()
    # for i, data in enumerate(openeds_loader):
    #     im, label = data
    #     im, label = np.asarray(np.squeeze(im.numpy(), axis = 0)), \
    #                 np.asarray(np.squeeze(label.numpy(), axis = 0))
    #
    #     ax = plt.subplot(1, 2, 1)
    #     ax.imshow(im, cmap = "gray")
    #     ax = plt.subplot(1, 2, 2)
    #     ax.imshow(label)
    #     plt.pause(1)
    #
    #     if i == 10:
    #         break
    # plt.title("OpenEDS")
    # plt.show()

    # fig = plt.figure()
    # for i, data in enumerate(calipso_loader):
    #     im, im_t, label = data
    #     im, im_t, label = np.asarray(np.squeeze(im.numpy(), axis=0)), \
    #                 np.asarray(np.squeeze(im_t.numpy(), axis=0)), \
    #                 np.asarray(np.squeeze(label.numpy(), axis=0))
    #
    #     ax = plt.subplot(1, 3, 1)
    #     ax.imshow(im, cmap = "gray")
    #     ax = plt.subplot(1, 3, 2)
    #     ax.imshow(im_t, cmap = "gray")
    #     ax = plt.subplot(1, 3, 3)
    #     ax.imshow(label)
    #     plt.pause(1)
    #
    #     if i == 10:
    #         break
    # plt.title("Calipso")
    # plt.show()
