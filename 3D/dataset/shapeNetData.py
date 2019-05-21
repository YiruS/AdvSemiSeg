"""
Code from https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/dataset.py
"""
from __future__ import print_function
import torch.utils.data as data
import os
import sys
import io

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
from PIL import Image

import torch
from torch.utils import data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


object_names = ['aeroplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp',
                'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
part_colors = [[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]]


def pts2img(pts, clr):
    def crop_image(image):
        image_data = np.asarray(image)
        assert (len(image_data.shape) == 3)
        image_data_bw = image_data.max(axis=2) if image_data.shape[-1] <= 3 else image_data[:, :, 3]
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
        return Image.fromarray(image_data_new)
    plt.close('all')
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)
    pts -= np.mean(pts,axis=0) #demean

    ax.view_init(20,200) # (20,30) for motor, (20, 20) for car, (20,210), (180, 60) for lamp, ax.view_init(20,110) for general, (180,60) for lamp, (20, 200) for mug and pistol
    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=clr,
        zdir='x',
        s=20,
        edgecolors=(0.5, 0.5, 0.5)  # (0.5,0.5,0.5)
    )

    ax.set_axis_off()
    ax.set_facecolor((1,1,1,0))
    ax.set_rasterized(True)
    ax.set_rasterization_zorder(1)
    ax.set_facecolor("white")
    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', transparent=True,
        bbox_inches='tight', pad_inches=0,
        rasterized=True,
        dpi=200
    )
    buf.seek(0)
    im = Image.open(buf)
    im = crop_image(im)
    buf.close()
    return im

def one_hot(label, num_classes):
    one_hot = np.zeros((1, num_classes), dtype = label.dtype)
    one_hot[0, label] = 1
    return one_hot

class ShapeNetGTDataset(data.Dataset):
    """
    Only generate seg labels.
    Creating a single dataloader to make it independent with the
    dataloader that outputs both images and seg labels.
    """
    def __init__(self,
                 num_classes,
                 npts=2048):
        self.num_classes = num_classes
        self.npts = npts

        hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
        self.filepath = os.path.join(hdf5_data_dir, "train_hdf5_file_list.txt")
        self.list =self.getDataFiles(self.filepath, hdf5_data_dir)
        pts, cls, seg = [], [], []
        for i in range(len(self.list)):
            p, c, s = self.load_h5_data_label_seg(self.list[i], self.npts)
            pts.append(p)
            cls.append(c)
            seg.append(s)

        self.pts = [pi for bp in pts for pi in bp]
        self.cls = [ci for bc in cls for ci in bc]
        self.seg = [si for bs in seg for si in bs]

        self.pts = np.asarray(self.pts, dtype=np.float32)
        self.cls = np.asarray(self.cls, dtype=np.int64)
        self.seg = np.asarray(self.seg, dtype=np.int64)


    def getDataFiles(self, list_filename, hdf5_data_dir):
        return [os.path.join(hdf5_data_dir, line.rstrip()) for line in open(list_filename)]

    def load_h5_data_label_seg(self, h5_filename, npts):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:,0:npts,:]
        label = f['label'][:]
        seg = f['pid'][:,0:npts]
        return data, label, seg

    def __getitem__(self, index):
        point = self.pts[index]
        cls = self.cls[index]
        cls_encoding = one_hot(cls, self.num_classes)
        seg = self.seg[index]

        point = torch.from_numpy(point)
        point = point.unsqueeze(0)
        # cls_encoding = cls_encoding.unsqueeze(0)
        cls_encoding = np.expand_dims(cls_encoding, axis=0)
        cls_encoding = torch.from_numpy(cls_encoding.astype(np.int64))
        seg = torch.from_numpy(seg.astype(np.int64))
        return point, cls_encoding, seg

    def __len__(self):
        return self.pts.shape[0]



class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 num_classes,
                 npts = 2048,
                 mode = "train",
                 aug_noise = False,
                 aug_rotate = False):
        self.num_classes = num_classes
        self.npts = npts
        self.mode = mode
        self.aug_noise = aug_noise
        self.aug_rotate = aug_rotate

        hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
        if self.mode == "train":
            self.filepath = os.path.join(hdf5_data_dir, "train_hdf5_file_list.txt")
        elif self.mode == "test":
            self.filepath = os.path.join(hdf5_data_dir, "test_hdf5_file_list.txt")
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        self.list =self.getDataFiles(self.filepath, hdf5_data_dir)
        pts, cls, seg = [], [], []
        for i in range(len(self.list)):
            p, c, s = self.load_h5_data_label_seg(self.list[i], self.npts)
            pts.append(p)
            cls.append(c)
            seg.append(s)

        self.pts = [pi for bp in pts for pi in bp]
        self.cls = [ci for bc in cls for ci in bc]
        self.seg = [si for bs in seg for si in bs]

        self.pts = np.asarray(self.pts, dtype=np.float32)
        self.cls = np.asarray(self.cls, dtype=np.int64)
        self.seg = np.asarray(self.seg, dtype=np.int64)


    def getDataFiles(self, list_filename, hdf5_data_dir):
        return [os.path.join(hdf5_data_dir, line.rstrip()) for line in open(list_filename)]

    def load_h5_data_label_seg(self, h5_filename, npts):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:,0:npts,:]
        label = f['label'][:]
        seg = f['pid'][:,0:npts]
        return data, label, seg

    def rotate_point_cloud(self, data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              Nx3 array, original batch of point clouds
            Return:
              Nx3 array, rotated batch of point clouds
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = data.copy()
        rotated_data = np.dot(shape_pc, rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original batch of point clouds
            Return:
              Nx3 array, jittered batch of point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    def __getitem__(self, index):
        point = self.pts[index]
        cls = self.cls[index]
        seg = self.seg[index]
        cls_encoding = one_hot(cls, self.num_classes)

        if self.aug_noise:
            point = self.jitter_point_cloud(point)
        if self.aug_rotate:
            point = self.rotate_point_cloud(point)

        point = torch.from_numpy(point)
        # point = point.unsqueeze(0)
        # cls_encoding = cls_encoding.unsqueeze(0) # 1x1xC
        # cls_encoding = np.expand_dims(cls_encoding, axis=0)
        cls_encoding = torch.from_numpy(cls_encoding.astype(np.int64))
        seg = torch.from_numpy(seg.astype(np.int64))
        return point, cls_encoding, seg

    def __len__(self):
        return self.pts.shape[0]

if __name__ == '__main__':
    shapenet = ShapeNetDataset(mode="train", num_classes=16)
    print("#train: {}".format(len(shapenet)))
    trainloader = data.DataLoader(shapenet, batch_size=1)
    for i, data in enumerate(trainloader):
        pts, cls, seg = data
        pts, cls, seg = pts.numpy().squeeze(), cls.numpy().squeeze(), seg.numpy().squeeze()
        if i<10:
            pnt_colormap_gt = []
            for j in range(seg.shape[0]):
                pnt_colormap_gt.append(part_colors[seg[j]])
            cls_idx = np.where(cls==1)[0][0]
            cur_obj = object_names[cls_idx]
            title = "pts_" + cur_obj + ".png"
            im = pts2img(pts, pnt_colormap_gt)
            im.save(os.path.join(BASE_DIR, title))
            print("Object ({:s})".format(cur_obj))
    # ps, cls, seg = data[0]
    # print("#pts: {}, type: {}".format(ps.size(), ps.type()))
    # print("class: {}, type: {}".format(cls.size(), cls.type()))
    # print("seg: {}, type: {}".format(seg.size(), seg.type()))

    shapenet = ShapeNetDataset(mode="test", num_classes=16)
    print("#test: {}".format(len(shapenet)))


