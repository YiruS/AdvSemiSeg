from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os,sys
import numpy as np
import pickle
import copy
import torch
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import torch.utils.data as data
file_path=os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/..'%file_path)
from utils.eval_utils import visualize_heatmap
from annotation.annotation_helper import AnnotationHelper

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    with Image.open(path) as f:
        f1 = f.convert('RGB').copy()
        return f1


def grayscale_loader(path):
    with Image.open(path) as f:
        return f.convert('L').copy()


def draw_gaussian_stride(img, pos, sigma, stride=1):
    start = stride / 2.0 - 0.5
    # draw gaussian mask around pos on img
    h, w = img.shape

    y, x = np.mgrid[0:h:1, 0:w:1]
    img[0:h, 0:w] = np.exp(
        -(start + x * stride - pos[0])**2 / (2.0 * sigma**2) -
        (start + y * stride - pos[1])**2 / (2.0 * sigma**2)
    )
    img[img < 0.01] = 0
    return img

def draw_gaussian_dots(img, pos, sigma, stride=1):
    # assumes pos is n x 2
    start = stride / 2.0 - 0.5
    # draw gaussian mask around pos on img
    h, w = img.shape

    y, x = np.mgrid[0:h:1, 0:w:1]
    for pi in range(pos.shape[0]):
        img[0:h, 0:w] += np.exp(
            -(start + x * stride - pos[pi,0])**2 / (2.0 * sigma**2) -
            (start + y * stride - pos[pi,1])**2 / (2.0 * sigma**2)
        )
    img[img < 0.01] = 0
    img = np.clip(img, 0., 1.)
    return img



def draw_gaussian_curve(img_size, pos, sigma, feat, stride=1):
    # fit a curve to the points, and draw a gaussian heatmap along the shape
    # first generate a image of original size
    orig_img_size = (img_size[0] * stride, img_size[1] * stride)

    img_flat = np.ones((np.prod(orig_img_size),), dtype=float)
    if feat == 'u_eyelid' or feat == 'b_eyelid':
        if np.all(pos < 0):
            return np.zeros(orig_img_size, dtype=float)

        # fit a curve
        z = np.polyfit(pos[:, 0], pos[:, 1], 3)
        func = np.poly1d(z)
        pos_w_new = np.arange(pos[:, 0].min(), pos[:, 0].max(), 1)
        pos_h_new = np.array([func(w) for w in pos_w_new])

        # only use points inside the image area to draw curves
        valid_idxs = np.where((pos_h_new.astype(int) >= 0)
                & (pos_h_new.astype(int) < orig_img_size[0])
                & (pos_w_new.astype(int) >= 0)
                & (pos_w_new.astype(int) < orig_img_size[1]))

        idxs = np.ravel_multi_index((
            pos_h_new[valid_idxs].astype(int),
            pos_w_new[valid_idxs].astype(int)),
            dims=orig_img_size)

        img_flat[idxs] = 0
        img = np.reshape(img_flat, orig_img_size)
    elif feat == 'e_iris' or feat == 'e_pupil':
        img = np.ones(orig_img_size, dtype=float)
        # fit an ellipse
        ellipse = cv2.fitEllipse(pos.astype('int'))
        cv2.ellipse(img, ellipse, 0)

    # convert img with binary map to distance map
    dist = distance_transform_edt(img)

    # convert distance map to gaussian kernel
    output = np.exp(-dist**2 / (2 * sigma**2))
    if stride > 1:
        output = output[int(stride / 2 - 1):-1:stride, int(stride / 2 - 1):-1:
                        stride]
    # set everything smaller than 0.01 to 0
    output[output < 0.01] = 0
    return output


def generate_seg_label(img_size, points_dict, stride=1):
    if ("u_eyelid" not in points_dict or
        "b_eyelid" not in points_dict or
        "e_iris" not in points_dict or
        "e_pupil" not in points_dict
        ):
        return np.zeros(img_size, dtype=float)
    else:
        orig_img_size = (img_size[0] * stride, img_size[1] * stride)
        label = np.zeros(orig_img_size, dtype='uint8')
        mask = np.zeros(orig_img_size, dtype='uint8')
        # generate segmentation label map

        # first fit two ellipses
        e1 = cv2.fitEllipse(points_dict['e_iris'].astype('int'))
        e2 = cv2.fitEllipse(points_dict['e_pupil'].astype('int'))
        # generate two poly for the two ellipses
        poly1 = cv2.ellipse2Poly(
            (int(e1[0][0]), int(e1[0][1])),
            (int(e1[1][0] / 2), int(e1[1][1] / 2)), int(e1[2]), 0, 360, 1
        )
        poly2 = cv2.ellipse2Poly(
            (int(e2[0][0]), int(e2[0][1])),
            (int(e2[1][0] / 2), int(e2[1][1] / 2)), int(e2[2]), 0, 360, 1
        )
        # generate one poly for eyelids
        sorted_u_eyelid = points_dict['u_eyelid']
        sorted_u_eyelid = sorted_u_eyelid[sorted_u_eyelid[:, 0].argsort()]
        sorted_b_eyelid = points_dict['b_eyelid']
        sorted_b_eyelid = sorted_b_eyelid[sorted_b_eyelid[:, 0].argsort()]
        poly3 = np.vstack([sorted_u_eyelid, sorted_b_eyelid[::-1]])

        # generate labels, mask is the eyelid mask
        if np.any(poly3 > 0):
            # only create eyelid mask if we have valid eyelid annotations
            cv2.fillPoly(mask, [np.array(poly3, dtype='int32')], 1)
            cv2.fillPoly(label, [np.array(poly3, dtype='int32')], 1)
        else:
            mask = np.ones(orig_img_size, dtype='uint8')
        cv2.fillPoly(label, [poly1], 2)
        cv2.fillPoly(label, [poly2], 3)
        label[mask == 0] = 0
        if stride > 1:
            label = label[int(stride / 2 - 1):-1:stride, int(stride / 2 - 1):-1:
                          stride]
        return label


def get_maximum_points(heatmaps):
    points = [
        np.unravel_index(part_map.argmax(), part_map.shape)
        for part_map in heatmaps
    ]
    return np.array(points)


class EyeHeatMapDataFromFolder(data.Dataset):
    def __init__(
        self,
        root,
        feat_list,
        heatmap_type='dot',
        heatmap_stride=2,
        sigma=20,
        loader=default_loader,
        dual_transform=None,
        include_valid_only=True,
        load_n=None,
        **kwargs
    ):
        self.root = root
        self.pkl_path = os.path.join(self.root, "labels")
        self.img_path = os.path.join(self.root, "images")
        self.heatmap_type = heatmap_type
        self.heatmap_stride = heatmap_stride
        self.dual_transform = dual_transform
        self.loader = loader
        self.sigma = sigma
        self.feat_list = feat_list
        self.include_valid_only = include_valid_only
        self.load_n = load_n

        # even if we want one output, still treat it as list
        if not type(self.heatmap_type) == list:
            self.heatmap_type = [self.heatmap_type]

        # if heatmap type is seg, then we need all features
        # to generate the segmentation labels
        if np.any([hmt is not None and 'seg' in hmt for hmt in self.heatmap_type]):
            self.feat_list = ['eyelid', 'e_iris', 'e_pupil']
        self.feat_dim = {
            "eyelid": 18,
            "e_iris": 8,
            "e_pupil": 8,
        }
        # go through all data in the folder,
        # find the ones with valid annotations for feat_list
        self.imgs, self.targets = self.get_data_from_folder()
        print('Found {} imgs and {} targets'.format(len(self.imgs), len(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def are_valid_points(self, points, img_shape=(740, 501)):
        points_valid = [
            (p[0] < img_shape[1] and p[1] >= 0
                and p[1] < img_shape[0] and p[0] > 0)
            for p in points
        ]
        return np.all(points_valid)

    def get_data_from_folder(self):
        images = []
        targets = []
        helper = AnnotationHelper()

        # if labels exist, load them along with corresponding images
        if os.path.isdir(self.pkl_path) \
        and len(os.listdir(self.pkl_path)) > 0:
            filenames = sorted(os.listdir(self.pkl_path))
            filenames = [f for f in filenames if f.endswith('.pkl')]

            if self.load_n is not None:
                skip = int(len(filenames) / self.load_n)
                filenames = filenames[::skip]
            print('Loading annotations from {} pkl files...'
                    .format(len(filenames)))

            for f in filenames:
                # get targets for feature points listed in feat_list
                # load filename
                pkl_file = open(os.path.join(self.pkl_path, f), 'rb')
                contents = pickle.load(pkl_file)
                points_all = {}
                # only save this data if all required features are annotated
                for feat in self.feat_list:
                    annotations = helper.get_kpset(contents, feat)
                    use_shape = (feat == 'e_pupil' or feat == 'e_iris')
                    if len(annotations) > 0:
                        points, shape = helper.get_points_per_annotation(
                            annotations[0], feat, use_shape
                        )
                    else:
                        points = -np.ones((self.feat_dim[feat], 2))
                    if feat == 'eyelid':
                        points_all['u_eyelid'] = points[0:9, :]
                        points_all['b_eyelid'] = points[9:, :]
                    else:
                        # for now, concatenate all points together.
                        # if a point is [-1, -1], it's invisible
                        points_all[feat] = points

                img_filename = os.path.join(self.img_path,
                        contents.image_filename)

                if self.include_valid_only:
                    img = cv2.imread(img_filename)
                    print(img_filename)
                    try:
                        points_valid_all = [
                            self.are_valid_points(points_all[key], img.shape)
                            for key in points_all
                        ]
                    except BaseException:
                        continue
                else:
                    points_valid_all = [True]

                if np.all(points_valid_all) and os.path.exists(img_filename):
                    images.append(img_filename)
                    targets.append(points_all)
            return images, targets
        else:
            # no labels exist, so just store image names
            filenames = sorted(os.listdir(self.img_path))
            if self.load_n is not None:
                skip = int(len(filenames) / self.load_n)
                filenames = filenames[::skip]
            images = [os.path.join(self.img_path, f)
                    for f in filenames if is_image_file(f) ]
            print('No pkl files found, loading {} images...'
                    .format(len(filenames)))
            return images, targets

    def __getitem__(self, index):
        img_filename = self.imgs[index]
        img_name = os.path.splitext(os.path.basename(img_filename))[0]
        img = self.loader(os.path.join(self.root, img_filename))
        orig_img = img.copy()
        # if we have targets, dual transform
        if len(self.targets) > 0:
            points_dict = copy.deepcopy(self.targets[index])
            # dual_transform worked on dictionary of points.
            if self.dual_transform is not None:
                img, points_dict = self.dual_transform(img, points_dict, t_type='dict')

        else:
            # if we do not have targets, do not try to transform
            # points
            if self.dual_transform is not None:
                img, points_dict = self.dual_transform(
                    img, None, t_type=None
                )


        # ++++++++++++++++++++++ generate heatmap +++++++++++++++++++++++++++++++
        # Note that img is already converted to torch tensor
        # after dual_transform function
        (c, h, w) = img.size()

        # size of the heatmap
        th = int(h / self.heatmap_stride)
        tw = int(w / self.heatmap_stride)

        targets = []
        for heatmap_type in self.heatmap_type:
            if heatmap_type is None:
                target = []
            elif heatmap_type == "dot":
                points = np.vstack(points_dict.values()).astype(int)
                # alternative way to generate heatmaps:
                # first scale points location to smaller size.
                # Then set stride = 1 in following function draw_gaussian_stride
                # points =
                # (np.vstack(points_dict.values())/self.heatmap_stride).astype(int)
                n_heatmaps = len(points)
                target = np.zeros((n_heatmaps, th, tw))
                for i in range(n_heatmaps):
                    target[i] = draw_gaussian_stride(
                        target[i],
                        points[i],
                        self.sigma,
                        stride=self.heatmap_stride
                    )
                target = torch.from_numpy(target).float()
            elif "curve" in heatmap_type:
                n_heatmaps = len(points_dict.keys())
                target = np.zeros((n_heatmaps, th, tw))
                for (i, feat) in enumerate(points_dict.keys()):
                    points = points_dict[feat].astype(int)
                    # alternative way to generate heatmaps:
                    # first scale points location to small size.
                    # Then set stride = 1 in following function draw_gaussian_curve
                    # points = (points_dict[feat]/self.heatmap_stride).astype(int)
                    # You can always replace this function with new functions
                    # that draw curve heatmaps differently
                    target[i] = draw_gaussian_curve(
                        (th, tw),
                        points,
                        self.sigma,
                        feat,
                        stride=self.heatmap_stride
                    )


                if "ellipsedots" in heatmap_type:
                    # instead of drawing ellipse, put dots at canonical locations
                    # do this for iris and pupil so the network doesnt get too confused

                    pupil_pts = points_dict['e_pupil']
                    # replace pupil ellipse with points
                    target[3] = draw_gaussian_dots(
                        np.zeros((th,tw)),
                        pupil_pts,
                        self.sigma,
                        stride=self.heatmap_stride
                        )

                    iris_pts = points_dict['e_iris']
                    # replace iris ellipse with points
                    target[2] = draw_gaussian_dots(
                        np.zeros((th,tw)),
                        iris_pts,
                        self.sigma,
                        stride=self.heatmap_stride
                        )

                # mask iris and pupil heatmaps using eyelid
                if 'mask' in heatmap_type:
                    within_eye_mask = np.ones((th, tw))
                    # h x w binary mask
                    eyelid_mask = np.ones((th, tw))
                    sorted_u_eyelid = points_dict['u_eyelid']
                    sorted_u_eyelid = sorted_u_eyelid[sorted_u_eyelid[:, 0].argsort()]
                    sorted_b_eyelid = points_dict['b_eyelid']
                    sorted_b_eyelid = sorted_b_eyelid[sorted_b_eyelid[:, 0].argsort()]

                    poly3 = np.vstack([sorted_u_eyelid, sorted_b_eyelid[::-1]])
                    # generate labels, mask is the eyelid mask
                    if np.any(poly3 > 0):
                        cv2.fillPoly(eyelid_mask, [np.array(poly3, dtype='int32')], 0)
                        within_eye_mask = 1-eyelid_mask.copy()
                    for c in [2, 3]:  # mask iris and pupil ellipse edges
                        target[c,:,:] = np.multiply(target[c,:,:], within_eye_mask)

                # e.g. "curve-pupildot-bkg" will load target with
                # 6 channels
                if "pupildot" in heatmap_type:
                    # put a dot at the pupil center
                    pupil_pts = points_dict['e_pupil'].astype(int)
                    pupil_ellipse = cv2.fitEllipse(pupil_pts)
                    target = np.concatenate([
                        target,
                        np.expand_dims(
                            draw_gaussian_stride(
                                np.zeros((th, tw)),
                                pupil_ellipse[0],
                                self.sigma,
                                stride=self.heatmap_stride
                            ), axis=0)], axis=0)
                if "pupildotonly" in heatmap_type:
                    target = np.expand_dims(target[-1], axis=0)
                elif "pupilellipseonly" in heatmap_type:
                    target = np.expand_dims(target[3], axis=0)
                elif "pupilblurredellipseonly" in heatmap_type:
                    # we want a filled ellipse with the edges blurred.
                    # this is not exactly a curve heatmap but not really
                    # a segmentation mask either.
                    pupil_pts = points_dict['e_pupil'].astype(int)
                    pupil_ellipse = cv2.fitEllipse(pupil_pts)
                    target = np.zeros((th,tw))
                    cv2.ellipse(target, pupil_ellipse, 1, -1)

                    target = cv2.GaussianBlur(target.astype(np.float32),
                            (int(self.sigma * 2 + 1), int(self.sigma * 2 + 1)),
                            self.sigma)
                    target = np.expand_dims(target, axis=0)

                if "bkg" in heatmap_type:
                    other_chans_sum = np.clip(
                            np.sum(target, axis=0, keepdims=True),
                            0, 1.0)
                    target = np.concatenate([
                        target,
                        1.0 - other_chans_sum
                        ], axis=0)


                target = torch.from_numpy(target).float()

            elif "seg" in heatmap_type:
                # for segmentation, target is numpy array of size (th, tw)
                # with pixel values representing the label index
                # since cross-entropy loss expects this format
                target = generate_seg_label(
                    (th, tw), points_dict, stride=self.heatmap_stride
                )

                if "seg-iris-pupil" in heatmap_type:
                    # only keep the iris and pupil masks
                    target[(not target == 3) & (not target == 2)] = 0
                    target[target == 2] = 1
                    target[target == 3] = 2
                elif "seg-pupil" in heatmap_type:
                    # only keep the pupil mask. make bkg the last channel
                    for label in range(3):
                        target[target == label] = 1
                    target[target == 3] = 0
                # the CrossEntropy2D loss takes a target of type torch.LongTensor
                target = torch.from_numpy(target).type(torch.LongTensor)
            # reconstruct a rotated version of input. useful for debugging
            # flow network
            elif heatmap_type == "recon-rot":
                # hack to insert rotation transform
                tforms = []
                for tform in self.dual_transform.transforms:
                    if not isinstance(tform, d_transforms.Normalize) \
                            and not isinstance(tform, d_transforms.ToTensor):
                        tforms.append(tform)
                tforms.append(d_transforms.RandomRotation(45.))
                tforms.append(d_transforms.ToTensor())
                tforms.append(d_transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
                tforms = d_transforms.Compose( tforms )
                rot_img, _ = tforms(
                    orig_img, points_dict, t_type='dict'
                )

                target = rot_img
            elif heatmap_type == "recon":
                # reconstruction, just return image
                target = img
            elif "smooth" in heatmap_type:  # flow smoothness reg, no label needed
                target = np.zeros((2, th, tw))
            else:
                print("type not defined!", self.heatmap_type)
                target = None
            targets.append(target)

        # for single task, this returns img, target, img_name as usual.
        # for multitask, this returns multiple targets in tuple of outputs
        return (img,) + tuple(targets) + (img_name,)

    def visualize_examples(self, vis_folder=None, random_samples=True):
        if random_samples:
            sampleslist=np.random.randint(0,self.__len__(),10)
        else:
            sampleslist=np.arange(self.__len__())
        for i in sampleslist:
            img, target,img_name = self.__getitem__(i)
            basename = os.path.basename(self.imgs[i])
            img, target = img.numpy().transpose((1, 2, 0)), target.numpy()
            # heatmap_result, point_result = visualize_heatmap(
            #     img, target, self.heatmap_type
            # )
            if vis_folder is not None:
                o=open("{}/{}".format(vis_folder, '%s.pkl'%basename.split('.')[0]),'wb')
                pickle.dump([img,target],o, protocol=2)
                o.close()
                # cv2.imwrite("{}/{}".format(vis_folder, basename), img)
                # cv2.imwrite("{}/target_{}".format(vis_folder, basename), target)

    def evaluate_metric(self, targets, output):
        if self.heatmap_type == "dot":
            target_points = [get_maximum_points(t) for t in targets]
            output_points = [get_maximum_points(o) for o in output]
            # Note here points[0] is h, points[1] is w
            points_dist = [np.mean(np.sqrt(((tp - op)**2).sum(axis=1)))
                           for (tp, op) in zip(target_points, output_points)]
            return np.mean(points_dist)
        else:
            # +++++++++++++++++++ NOT IMPLEMENTED YET! ++++++++++++++++++++++++++++++++
            return 0
