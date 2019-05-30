from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
import numbers
import torchvision.transforms as t_transforms


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pos=None, t_type='pos'):
        for t in self.transforms:
            if pos is None:
                img = t(img)
                return img
            else:
                img, pos = t(img, pos, t_type)
                return img, pos



class ToTensor(object):
    def __call__(self, img, pos, t_type='pos'):
        return t_transforms.ToTensor()(img), pos


class Normalize(object):
    """ Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, pos, t_type='pos'):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, pos

class ChangeBrightness(object):
    """
    Change the brightness of input .npy given 'scale'.
    'scale' = 0, return with black image.
    'scale' = 1, return original image.
    'scale' = 2, double the brightness.
    """
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, img):
        if np.max(img) <= 1.0:
            im_arr_int = cv2.normalize(
                img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype = cv2.CV_32F
            )
        else:
            im_arr_int = img.copy()

        im_arr_int = im_arr_int.astype(np.uint8)
        im_pil = Image.fromarray(im_arr_int)
        enhancer = ImageEnhance.Brightness(im_pil)
        enhanced_image = enhancer.enhance(self.scale)
        im_arr_float = np.asarray(enhanced_image)
        im_arr_float = im_arr_float.astype(np.float) / 255.0

        return im_arr_float



class Scale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, pos, t_type='pos'):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, pos
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            scale = 1.0 * self.size / w
        else:
            oh = self.size
            ow = int(self.size * w / h)
            scale = 1.0 * self.size / h

        if t_type == 'pos':
            pos = pos * scale
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key] *= scale
        elif t_type == 'ellipse':
            for i in range(4):
                pos[i] = pos[i] * scale
        return img.resize((ow, oh),Image.BILINEAR), pos ### change to bilinear


class ScaleByFactor(object):
    """ Rescales the input PIL.Image by the given
    factor.
    scale_factor: scaling factor (int or float)
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, scale_factor, interpolation=Image.BILINEAR):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def __call__(self, img, pos, t_type='pos'):
        w, h = img.size
        if self.scale_factor == 1:
            return img, pos
        w, h = img.size
        ow, oh = int(w * self.scale_factor), int(h * self.scale_factor)

        if t_type == 'pos':
            pos = pos * self.scale_factor
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key] *= self.scale_factor
        elif t_type == 'ellipse':
            for i in range(4):
                pos[i] = pos[i] * self.scale_factor
        return img.resize((ow, oh)), pos


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, pos, t_type='pos'):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        if t_type == 'pos':
            pos[:, 0] -= x1
            pos[:, 1] -= y1
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key][:, 0] -= x1
                pos[key][:, 1] -= y1
        elif t_type == 'ellipse':
            pos[0] -= x1
            pos[1] -= y1
        return img.crop((x1, y1, x1 + tw, y1 + th)), pos


class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        if not isinstance(padding, tuple):
            assert isinstance(padding, numbers.Number)
            padding = (padding, padding)
        assert isinstance(fill, numbers.Number)
        self.padding = padding
        self.fill = fill

    def __call__(self, img, pos, t_type='pos'):
        if t_type == 'pos':
            pos[:, 0] += self.padding[0]
            pos[:, 1] += self.padding[1]
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key][:, 0] += self.padding[0]
                pos[key][:, 1] += self.padding[1]
        elif t_type == 'ellipse':
            pos[0] += self.padding[0]
            pos[1] += self.padding[1]

        if self.padding[0] == self.padding[1]:
            return ImageOps.expand(img, border=self.padding[0], fill=self.fill), pos
        else:
            # convert to numpy to do unequal h,w padding
            img_np = np.asarray(img)
            if len(img_np.shape)==3:
                padding_tuple = (
                        (self.padding[1], self.padding[1]),
                        (self.padding[0], self.padding[0]),
                        (0, 0)
                        )
            else:
                padding_tuple = (
                        (self.padding[1], self.padding[1]),
                        (self.padding[0], self.padding[0])
                        )
            img_np = np.pad(
                    img_np, padding_tuple,
                    mode='constant', constant_values=self.fill
                    )
            return Image.fromarray(img_np.astype(np.uint8)), pos



class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, pos, t_type='pos'):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, pos

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if t_type == 'pos':
            pos[:, 0] -= x1
            pos[:, 1] -= y1
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key][:, 0] -= x1
                pos[key][:, 1] -= y1
        elif t_type == 'ellipse':
            pos[0] -= x1
            pos[1] -= y1
        return img.crop((x1, y1, x1 + tw, y1 + th)), pos


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, pos, t_type='pos'):
        if random.random() < 0.5:
            if t_type == 'pos':
                pos[:, 0] = img.size[0] - 1 - pos[:, 0]
            elif t_type == 'dict':
                for key in pos.keys():
                    pos[key][:, 0] = img.size[0] - 1 - pos[:, 0]
            elif t_type == 'ellipse':
                pos[0] = img.size[0] - pos[0]
            return img.transpose(Image.FLIP_LEFT_RIGHT), pos
        else:
            return img, pos



# TODO: get randomrotation to call rotation, lots of repeated functionality
class RandomRotation(object):
    """Randomly rotates the given PIL.Image by a value in (-max_rot, max_rot)
    """
    def __init__(self, max_rot=30.):
        self.max_rot = max_rot

    def __call__(self, img, pos, t_type='pos'):
        rot_deg = random.uniform( -self.max_rot, self.max_rot)
        rot_rad = rot_deg * np.pi / 180.
        w, h = img.size
        img_center = np.asarray([w/2.0, h/2.0])

        rot_matrix = np.asarray([[np.cos(rot_rad), -np.sin(rot_rad)],
            [np.sin(rot_rad), np.cos(rot_rad)]])
        if t_type == 'pos':
            for pi in range(pos.shape[0]):
                pos[pi,:] = pos[pi,:] - img_center
                pos[pi,:] = np.matmul(pos[pi,:], rot_matrix)
                pos[pi,:] = pos[pi,:] + img_center
        elif t_type == 'dict':
            for key in pos.keys():
                for pi in range(pos[key].shape[0]):
                    pos[key][pi,:] = pos[key][pi,:] - img_center
                    pos[key][pi,:] = np.matmul(pos[key][pi,:], rot_matrix)
                    pos[key][pi,:] = pos[key][pi,:] + img_center
        elif t_type == 'ellipse':
            pos[0] = np.asarray(pos[0]) - img_center
            pos[0] = np.matmul(pos[0], rot_matrix)
            pos[0] = tuple((pos[0] + img_center).tolist())
            pos[-1] += rot_deg

        return img.rotate(rot_deg), pos


class Rotation(object):
    """Rotates the given PIL.Image by the specified number of degrees
    """
    def __init__(self, rot_deg=30.):
        self.rot_deg = rot_deg
        self.rot_rad = rot_deg * np.pi / 180.
        self.rot_matrix = np.asarray([[np.cos(self.rot_rad), -np.sin(self.rot_rad)],
            [np.sin(self.rot_rad), np.cos(self.rot_rad)]])

    def __call__(self, img, pos, t_type='pos'):
        w, h = img.size
        img_center = np.asarray([w/2.0, h/2.0])

        if t_type == 'pos':
            for pi in range(pos.shape[0]):
                pos[pi,:] = pos[pi,:] - img_center
                pos[pi,:] = np.matmul(pos[pi,:], self.rot_matrix)
                pos[pi,:] = pos[pi,:] + img_center
        elif t_type == 'dict':
            for key in pos.keys():
                for pi in range(pos[key].shape[0]):
                    pos[key][pi,:] = pos[key][pi,:] - img_center
                    pos[key][pi,:] = np.matmul(pos[key][pi,:], self.rot_matrix)
                    pos[key][pi,:] = pos[key][pi,:] + img_center
        elif t_type == 'ellipse':
            pos[0] = np.asarray(pos[0]) - img_center
            pos[0] = np.matmul(pos[0], self.rot_matrix)
            pos[0] = tuple((pos[0] + img_center).tolist())
            pos[-1] += rot_deg

        return img.rotate(self.rot_deg), pos


class RandomIntensityScale(object):
    def __init__(self, scale_factor, scale_channels_independently=True):
        if isinstance(scale_factor, numbers.Number):
            # if a single scale factor is provided, randomize symetrically
            self.scale_factor = (1.0 - scale_factor, 1.0 + scale_factor)
        else:
            self.scale_factor = scale_factor
        self.scale_channels_independently = scale_channels_independently


    def __call__(self, img, pos, t_type='pos'):
        img_np = np.asarray(img).astype(np.float32)
        if self.scale_channels_independently \
        and len(img_np.shape) > 2 and img_np.shape[-1] > 1:
            n_channels = img_np.shape[-1]
            scale_factors = [random.uniform(self.scale_factor[0], self.scale_factor[1])
                    for _ in range(n_channels)]
            scale_factors = np.tile(
                np.reshape(scale_factors,
                    [1]*(len(img_np.shape)-1) + [n_channels]),
                img_np.shape[:-1] + (1,))
            img_np = np.multiply(img_np, scale_factors)
        else:
            img_np = img_np * random.uniform(self.scale_factor[0], self.scale_factor[1])
        img_np = np.clip(img_np, 0, 255)
        img = Image.fromarray(img_np.astype(np.uint8))
        return img, pos


class RandomScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, pos, t_type='pos'):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, pos
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            scale = 1.0 * self.size / w
        else:
            oh = self.size
            ow = int(self.size * w / h)
            scale = 1.0 * self.size / h

        if t_type == 'pos':
            pos = pos * scale
        elif t_type == 'dict':
            for key in pos.keys():
                pos[key] *= scale
        elif t_type == 'ellipse':
            for i in range(4):
                pos[i] = pos[i] * scale
        return img.resize((ow, oh)), pos
