from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import torch

import torch.onnx

from collections import Iterable


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
sys.path.append("%s/../" % file_path)
from helper_functions.torchsummary import summary

NUM_CLASSES = 4

def _create_inputs(input_or_shape):
    if isinstance(input_or_shape, torch.Tensor):
        inputs = input_or_shape.cpu()
        inputs.requires_grad_()
    elif isinstance(input_or_shape, Iterable):
        inputs = torch.randn(*input_or_shape)
    else:
        raise ValueError(
            "Cannot recognize the argument type " + str(type(input_or_shape))
        )
    return inputs

def load_display_model(args):
    '''
    module to load and display pytorch model network
    :param args loaded through the config .ini files
    # see for example ./Ini_Files
    '''
    if args.network == 'segnet_small':
        from models.SegNet import SegNet_Small
        model = SegNet_Small(
            args.channels,
            args.classes,
            args.skip_type,
            BR_bool = args.BR,
            separable_conv = args.SC
        )
        if args.CUDA:
            model = model.cuda(args.gpu)

    summary(
        model,
        (args.channels, args.image_size, args.image_size),
        args
    )

    try:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
    except Exception as e:
        print (e)
        sys.exit(0)

    return model

def load_models(mode, device, args):
    """

    :param mode: "SS" or "Discriminator"
    :param args:
    :return:
    """

    if mode == "SS":
        if args.network == "segnet_small":
            from models.SegNet import SegNet_Small
            model = SegNet_Small(
                args.channels,
                args.classes,
                args.skip_type,
                BR_bool=args.BR,
                separable_conv=args.SC
            )
            model = model.to(device)

        summary(
            model,
            (args.channels, args.image_size, args.image_size),
            args
        )
    elif mode == "Discriminator":
        from models.discriminator import FCDiscriminator
        model = FCDiscriminator(
            num_classes = NUM_CLASSES,
        )
    else:
        raise ValueError("Invalid mode {}!".format(mode))

    try:
        if args.checkpoint_SS:
            model.load_state_dict(torch.load(args.checkpoint_SS))
        if args.checkpoint_DNet:
            model.load_state_dict(torch.load(args.checkpoint_DNet))

    except Exception as e:
        print (e)
        sys.exit(0)

    return model