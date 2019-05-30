from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import sys
import os
import torch
import time
import torch.nn as nn
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)
from helper_functions.torchsummary import summary, initialize_weights
from torchvision import models
F = nn.functional

def load_pretrained_vgg(model, vgg_model):
    '''
    module to load pretrained_vgg model weights
    '''
    count_conv=0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            count_conv+=1
    print (count_conv)
    count=0
    for m in model.modules():
        for v in vgg_model.modules():
            if isinstance(m,nn.Conv2d) and isinstance(v,nn.Conv2d) and count<count_conv:
                if m.weight.size()==v.weight.size():
                    m.weight.data=v.weight.data
                    m.bias.data=v.bias.data
        if isinstance(m,nn.Conv2d):
            count+=1
    return model


### define layers:



class _EncoderBlock(nn.Module):
    '''
    Encoder block structured aroung vgg_model blocks
    '''
    def __init__(self, in_channels, out_channels, kernel_size, separable_conv=False,name='default',BN=True):
        super(_EncoderBlock, self).__init__()
        self.name=name

        _encoder_layer_SC_WBN=[
        nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        ]

        _encoder_layer_SC_NBN=[
        nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        ]

        if not separable_conv:
            layers = [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            if BN:
                layers=_encoder_layer_SC_WBN
            else:
                layers=_encoder_layer_SC_NBN
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    '''
    Decoder blocks using Transpose Convolution blocks
    '''
    def __init__(self, in_channels, out_channels, kernel_size,is_nonlinear=True,separable_conv=False,name='default',BN=True):
        super(_DecoderBlock, self).__init__()
        self.name=name
        _decoder_layer_SC_WBN=[
        nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
        nn.BatchNorm2d(out_channels),]

        _decoder_layer_SC_NBN=[
        nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=1,padding=1,groups=in_channels),
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),]

        if not separable_conv:
            layers = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            if BN:
                layers=_decoder_layer_SC_WBN
            else:
                layers=_decoder_layer_SC_NBN

        if is_nonlinear:
            layers.append(nn.ReLU(inplace=True))

        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

class BR(nn.Module):
    '''
    Boundry refinement block
    See: https://arxiv.org/pdf/1703.02719.pdf
    '''
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)

    def forward(self,x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        return x

class SegNet_Small(nn.Module):
    '''
    Low complexity version of SegNet Semantic Segmentation model
    Designed for eye feature segmentation task
    '''
    def __init__(self, input_channels, num_classes,skip_type=None,BR_bool=False,separable_conv=False,caffe=False,mode='nearest',BN=True):
        super(SegNet_Small, self).__init__()
        self.BR_bool=BR_bool
        self.skip_type=skip_type
        self.caffe=caffe
        self.mode=mode
        self.BN=BN
        self.enc10 = _EncoderBlock(input_channels, 64, 3,separable_conv=separable_conv,name='enc10',BN=self.BN)
        self.enc11 = _EncoderBlock(64, 64, 3,separable_conv=separable_conv,name='enc11',BN=self.BN)
        self.enc20 = _EncoderBlock(64, 128, 3,separable_conv=separable_conv,name='enc20',BN=self.BN)
        self.enc21 = _EncoderBlock(128, 128, 3,separable_conv=separable_conv,name='enc21',BN=self.BN)
        self.enc30 = _EncoderBlock(128, 256, 3,separable_conv=separable_conv,name='enc30',BN=self.BN)
        self.enc31 = _EncoderBlock(256, 256, 3,separable_conv=separable_conv,name='enc31',BN=self.BN)
        self.enc32 = _EncoderBlock(256, 256, 3,separable_conv=separable_conv,name='enc32',BN=self.BN)

        self.dec32 = _DecoderBlock(256, 256, 3,separable_conv=separable_conv,name='dec32',BN=self.BN)
        self.dec31 = _DecoderBlock(256, 256, 3,separable_conv=separable_conv,name='dec31',BN=self.BN)
        self.dec30 = _DecoderBlock(256, 128, 3,separable_conv=separable_conv,name='dec30',BN=self.BN)
        self.dec21 = _DecoderBlock(128, 128, 3,separable_conv=separable_conv,name='dec21',BN=self.BN)
        self.dec20 = _DecoderBlock(128, 64, 3,separable_conv=separable_conv,name='dec20',BN=self.BN)
        self.dec11 = _DecoderBlock(64, 64, 3,separable_conv=separable_conv,name='dec11',BN=self.BN)
        self.dec10 = _DecoderBlock(64, num_classes, 3,is_nonlinear=False,separable_conv=separable_conv,name='dec10',BN=self.BN)
        if self.BR_bool:
            self.BR=BR(num_classes)
        initialize_weights(self.enc10,self.enc11,self.enc20,self.enc21,self.enc30,\
        self.enc31,self.enc32,self.dec32,self.dec31,self.dec30,self.dec21,self.dec20,\
        self.dec11,self.dec10)

    def forward(self, x):
        dim_0 = x.size()

        enc1 = self.enc10(x)
        enc1 = self.enc11(enc1)
        x_1, indices_1 = F.max_pool2d(
            enc1, kernel_size=2, stride=2, return_indices=True
        )

        dim_1 = x_1.size()
        enc2 = self.enc20(x_1)
        enc2 = self.enc21(enc2)
        x_2, indices_2 = F.max_pool2d(
            enc2, kernel_size=2, stride=2, return_indices=True
        )

        dim_2 = x_2.size()
        enc3 = self.enc30(x_2)
        enc3 = self.enc31(enc3)
        enc3 = self.enc32(enc3)
        x_3, indices_3 = F.max_pool2d(
            enc3, kernel_size=2, stride=2, return_indices=True
        )

        if self.caffe:
            dec3=nn.functional.interpolate(x_3,scale_factor=2, mode=self.mode)
        else:
            dec3 = F.max_unpool2d(
                x_3, indices_3, kernel_size=2, stride=2, output_size=dim_2
            )
        dec3 = self.dec32(dec3)
        dec3 = self.dec31(dec3)
        dec3 = self.dec30(dec3)

        if self.caffe:
            dec2=nn.functional.interpolate(dec3,scale_factor=2, mode=self.mode)
        else:
            dec2 = F.max_unpool2d(
                dec3, indices_2, kernel_size=2, stride=2, output_size=dim_1
            )
        if self.skip_type is not None:
            dec2+=enc2
        dec2 = self.dec21(dec2)
        dec2 = self.dec20(dec2)
        if self.caffe:
            dec1=nn.functional.interpolate(dec2,scale_factor=2, mode=self.mode)
        else:
            dec1 = F.max_unpool2d(
                dec2, indices_1, kernel_size=2, stride=2, output_size=dim_0
            )
        if self.skip_type is not None:
            if 'mul' in self.skip_type.lower():
                dec1*=enc1
            if 'add' in self.skip_type.lower():
                dec1+=enc1
        dec1 = self.dec11(dec1)
        dec1 = self.dec10(dec1)
        if self.BR_bool:
            dec1=self.BR(dec1)
        if self.caffe:
            return dec1
        else:
            dec_softmax = F.softmax(dec1, dim=1)
            return dec1, dec_softmax


def parse_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))


if __name__ == "__main__":
    '''
    Run time script to diplay model architecture and information on model
    complexity/size
    '''
    parser = optparse.OptionParser()
    parser.add_option(
        "--image-size",
        type="str",
        action="callback",
        callback=parse_list,
        dest="image_size",
    )
    parser.add_option(
        "--num-classes", help="num classes", dest="num_classes", type=int, default=4
    )
    parser.add_option(
        "--gpu", help="gpu number", dest="gpu", type=int, default=0
    )
    (opts, args) = parser.parse_args()
    print (opts,args)
    if opts.image_size is None:
        img_size = [3, 360, 480]
    else:
        img_size = map(int, opts.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #x=torch.rand(32,3,128,128).cuda(opts.gpu)
    model_noSC = SegNet_Small(3,opts.num_classes, BR_bool=False,separable_conv=False).to(device).cuda(opts.gpu)
    summary(model_noSC, tuple(img_size) , opts)
    #model_noSC.forward(x)
    print ('\n')
    model_SC = SegNet_Small(3,opts.num_classes, BR_bool=False,separable_conv=True).to(device).cuda(opts.gpu)
    #model_SC.forward(x)
    #vgg=models.vgg16(pretrained=True).cuda(opts.gpu)
    summary(model_SC, tuple(img_size) , opts)
