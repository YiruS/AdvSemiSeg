from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import sys

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as t_transforms

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)
import utils.photometric_transforms as ph_transforms
from utils.generic_utils import id_generator, memory, get_free_gpu

import matplotlib.pyplot as plt
import copy
plt.switch_backend('agg')

CUDA = torch.cuda.is_available()
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 4

TRANSFORM_SET = [
    ph_transforms.ChangeBrightness(2.0)
]

def generate_result_images(
        input_image,
        target_image,
        pred_image,
        args,
        iou,
        count=0,
):
    '''
    helper module to generate figure showing input image/predicted mask and optinally
    ground truth segmentation mask
    :param input_image: 3D-array of input image
    :param target_image: 2D-array of target image
    :param pred_image: 2D-array of pred_image
    :param args: input arguments from __main__
    :param iou: IOU for the prediction
    :param count: counter to keep tab on the image count to generate prediction output file
    '''
    images_dir = "%s/Test_Images" % args.output_dir
    try:
        if not os.path.isdir(images_dir):
            os.mkdir(images_dir)

        fig = plt.figure()
        if target_image is None:
            a = fig.add_subplot(1,2,1)
        else:
            a = fig.add_subplot(1,3,1)
        c, h, w = input_image.shape
        if c == 1:
            input_image = input_image.reshape(h, w)
        else:
            input_image = np.transpose(input_image, (2,1,0))
            input_image = np.transpose(input_image, (1,0,2))
        plt.imshow(input_image, cmap="gray")
        a.set_title("Input Image")

        if target_image is not None:
            a = fig.add_subplot(1,3,2)
            plt.imshow(target_image)
            a.set_title("GT")

        if target_image is None:
            a = fig.add_subplot(1,2,2)
        else:
            a = fig.add_subplot(1,3,3)
        plt.imshow(pred_image)
        a.set_title('Prediction w/ IoU {:.3f}'.format(iou))
        fig.savefig(os.path.join(images_dir, "prediction_{}.png".format(count)))
        plt.close(fig)

    except Exception as e:
        print (e)
        sys.exit(0)

def generate_segmented_image(
        model,
        imagepath,
        image_size=184,
        filter=10,
        cuda=CUDA,
        outputfile=None,
        gen_features=False,
        return_output=False
):
    '''
    Utility function to generate segmented images from a pre-trained feature segmentation model
    # currently defaults to SegNet_Small model, want to generalize
    :param model: pytorch model (already loaded into memory)
    :param imagepath: path to input image file
    :param image_size: image size (currently support a scalar, assuming image scaling to a sqaure output image)
    :param filter: filter number to observe
    :param cuda: Boolean variable indicating gpu availability
    :param outputfile: path to outputfile where the result will be saved
    :param gen_features: Boolean indicator to determine whether want to see filter output
    :param return_output: Boolean to get the results out
    :return:
    '''
    output_features=[]
    model.eval()
    transforms = copy.copy(TRANSFORM_SET)
    # transforms.append(d_transforms.Scale(image_size))
    photometric_transform = ph_transforms.Compose(transforms)

    ### convert .npy to PIL format ###
    img = np.load(imagepath)
    if np.max(img) <= 1.0:
        im_arr_int = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
    else:
        im_arr_int = img.copy()

    im_arr_int = im_arr_int.astype(np.uint8)
    im_pil = Image.fromarray(im_arr_int)
    ### convert .npy to PIL format ###

    img = photometric_transform(im_pil)
    imgT = t_transforms.ToTensor()(img)
    (c, w, h) = imgT.size()
    if cuda:
        imgT = imgT.reshape(-1, c, w, h).cuda(0)
    input_tensor = torch.autograd.Variable(imgT)
    with torch.set_grad_enabled(False):
        predicted_tensor, softmaxed_tensor = model(input_tensor)
        x = input_tensor
        for k in model._modules.keys():
            x = model._modules[k](x)
            output_features.append(x)
    predicted_mx = softmaxed_tensor[0].detach().cpu().numpy()

    keys = list(model._modules.keys())
    if gen_features:
        fig_x_dim = len(keys)/5
        fig_y_dim = 5
    else:
        fig_x_dim = 1
        fig_y_dim = 2
    fig = plt.figure()
    a = fig.add_subplot(fig_x_dim,fig_y_dim,1)
    plt.imshow(input_tensor[0].detach().cpu().transpose(1,2).transpose(0, 2))
    a.set_title('Input Image')

    a = fig.add_subplot(fig_x_dim,fig_y_dim,2)
    predicted_mx = softmaxed_tensor[0].detach().cpu().numpy()
    predicted_mx = predicted_mx.argmax(axis=0)
    plt.imshow(predicted_mx)
    a.set_title('Predicted Mask')

    if gen_features:
        count=2
        for i in range(fig_x_dim):
            for j in range(fig_y_dim):
                a = fig.add_subplot(fig_x_dim,fig_y_dim,count+1)
                of = output_features[count-2][0].detach().cpu().numpy()[filter,:,:]
                print (of.shape)
                plt.imshow(of)
                a.set_title(keys[count-2])
                count+=1
    if outputfile is None:
        outputfolder='%s/nfs/Temp'%os.environ['HOME']
        if not os.path.isdir(outputfolder):
            os.mkdir(outputfolder)
        outputfile='%s/output%s.png'%(outputfolder,id_generator())

    fig.savefig(outputfile)
    plt.close(fig)
    if return_output:
        return img, predicted_mx,output_features
