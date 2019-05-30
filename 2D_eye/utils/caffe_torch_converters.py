# convert caffe model to pytorch model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict


class CaffeTorchConverter():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def get_caffe_model(self):
        return self.caffe_net

    def get_caffe_param(self, layer_name, param_name):
        if param_name == 'weight':
            k = self.caffe_net.params[layer_name][0].data
        elif param_name == 'bias':
            k = self.caffe_net.params[layer_name][1].data
        elif param_name == 'running_mean':
            k = self.caffe_net.params[layer_name][0].data / \
                self.caffe_net.params[layer_name][2].data
        elif param_name == 'running_var':
            k = self.caffe_net.params[layer_name][1].data / \
                self.caffe_net.params[layer_name][2].data
        return k

    def set_caffe_param(self, layer_name, param_name, data):
        if param_name == 'weight':
            assert self.caffe_net.params[layer_name][0].data.shape == data.shape
            self.caffe_net.params[layer_name][0].data[...] = data
        elif param_name == 'bias':
            assert self.caffe_net.params[layer_name][1].data.shape == data.shape
            self.caffe_net.params[layer_name][1].data[...] = data
        elif param_name == 'running_mean':
            self.caffe_net.params[layer_name][0].data[...] = data
            self.caffe_net.params[layer_name][2].data[...] = np.array([1.0])
        elif param_name == 'running_var':
            self.caffe_net.params[layer_name][1].data[...] = data

    def parse_pth_varname(self, layer_name_map, pth_varname):
        idx = pth_varname.rfind('.')
        param_name = pth_varname[idx + 1:]
        pth_layer_name = pth_varname[0:idx]
        caffe_layer_name = layer_name_map[pth_layer_name]
        print(pth_varname, caffe_layer_name, param_name)
        return caffe_layer_name, param_name

    def convert_to_pth(self, pth_model, layer_name_map, strict_mode=True):
        new_state_dict = OrderedDict()

        for var_name in pth_model.state_dict():
            caffe_layer_name, param_name = self.parse_pth_varname(
                layer_name_map, var_name
            )
            w2 = self.get_caffe_param(caffe_layer_name, param_name)
            w2 = torch.from_numpy(w2).float()
            w1 = pth_model.state_dict()[var_name]
            if w1.size() != w2.size():
                print(w1.size(), w2.size())
                if strict_mode:
                    assert w1.size() == w2.size()
                else:
                    new_state_dict[var_name] = w1
                    continue
            new_state_dict[var_name] = w2
        pth_model.load_state_dict(new_state_dict)
        return pth_model

    def convert_from_pth(self, pth_model, layer_name_map):
        for var_name in pth_model.state_dict():
            caffe_layer_name, param_name = self.parse_pth_varname(
                layer_name_map, var_name
            )
            print(var_name)
            w_pth = pth_model.state_dict()[var_name].numpy()
            self.set_caffe_param(caffe_layer_name, param_name, w_pth)

    def dist_(self, caffe_tensor, th_tensor):
        t = th_tensor[0]
        c = caffe_tensor[0]
        print("t.shape", t.shape)
        print("c.shape", c.shape)

        d = np.linalg.norm(t - c)
        print("d", d)

    def verify(self, caffe_model, pth_model, layer_name_map, img):
        # the image has to be in numpy format (for example, read by opencv)
        # first normalize to [-0.5, 0.5]
        img = img / 256.0 - 0.5
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        # input needs to be C X H X W for both caffe and pytorch models
        caffe_model.blobs['data'].data[0] = img.transpose((2, 0, 1))
        caffe_model.forward()

        pth_model.eval()
        o = []

        def hook(module, input, output):
            tmp = output.data.numpy()
            o.append(tmp)

        pth_modules = pth_model._modules
        used_layer = []
        # add hooks to record output from every convolutional layer of pytorch model
        for i in range(0, len(pth_modules), 1):
            pth_layer_name = pth_modules.keys()[i]
            if pth_layer_name in layer_name_map and isinstance(
                pth_modules[pth_layer_name], nn.Conv2d
            ):
                pth_modules[pth_layer_name].register_forward_hook(hook)
                used_layer.append(pth_layer_name)
        print(used_layer)
        output_stage1, output_final = pth_model(
            Variable(
                torch.from_numpy(img[np.newaxis, :].transpose(0, 3, 1, 2))
                .float(),
                volatile=True
            )
        )

        # compare output from every concolutional layer of caffe model and pytorch model
        for o_idx, layer_name in enumerate(used_layer):
            caffe_layer_name = layer_name_map[layer_name]
            print("Comparing layer: ", layer_name)
            self.dist_(caffe_model.blobs[caffe_layer_name].data, o[o_idx])

        # get output
        caffe_o = caffe_model.blobs[caffe_model.outputs[0]].data[0]
        pth_o = output_final.data[0].numpy()
        self.dist_(caffe_o, pth_o)
        return caffe_o, pth_o
