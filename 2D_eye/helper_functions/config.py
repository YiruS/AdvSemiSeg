from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys,getopt
import configparser
import re
import glob


def parse_datalist(data_list):
    dataroot_list = [str(d) for d in data_list.split(',')]
    return dataroot_list

def ReadConfig(File):
    '''
    Module that defines the structure for model config files
    See for example ./Ini_Files
    :param File: .ini file containing information on network configuration
    See for example ../Ini_Files
    '''
    config=configparser.ConfigParser()
    config.read(File)
    section=config.sections()
    Global_Params={}
    Global_Params['g_Network']=eval(config.get(section[0],'g_network'))
    Global_Params['g_Channels']=eval(config.get(section[0],'g_channels'))
    Global_Params['g_Classes']=eval(config.get(section[0],'g_classes'))
    Global_Params['g_dataset']=eval(config.get(section[0],'g_dataset'))

    Train_Params={}
    Train_Params['LR']=eval(config.get(section[1],'lr'))
    Train_Params['Batch_Size']=eval(config.get(section[1],'batch_size'))
    Train_Params['Loss']=eval(config.get(section[1],'loss'))
    Train_Params['Workers']=eval(config.get(section[1],'workers'))
    Train_Params['Epochs']=eval(config.get(section[1],'num_epochs'))
    Train_Params['L2']=eval(config.get(section[1],'l2'))
    Train_Params['brightness_scale'] = eval(config.get(section[1], 'brightness_scale'))

    return Global_Params, Train_Params

def get_net_config(GP):
    '''
    helper module to read the global model parameters that define the network
    :param GP: Global_Params from .ini file
    '''
    skip_type=None; BR_bool=False; SC_bool=False; BN_bool=True
    if len(GP)==4:
        network=GP[0];skip_type=GP[1];BR_bool=GP[2];BN_bool=GP[3];SC_bool=True
    if len(GP)==3:
        network=GP[0];skip_type=GP[1];BR_bool=True;SC_bool=False
    if len(GP)==2:
        network=GP[0];skip_type=GP[1];BR_bool=False;SC_bool=False
    if len(GP)==1:
        network=GP[0];skip_type='no_skip';BR_bool=False;SC_bool=False
    return network, skip_type,BR_bool,SC_bool,BN_bool

def augment_args(GP,TP,args):
    '''
    helper module to augment input arguments for __main__
    :param GP: Global_Params from .ini file
    :param TP: Train_Params from .ini file
    :param args: input arguments from __main__
    '''
    args.network, args.skip_type, args.BR, args.SC,args.BN =get_net_config(GP['g_Network'])
    args.channels=GP['g_Channels']
    args.classes=GP['g_Classes']
    args.dataset = GP['g_dataset']
    args.lr=TP['LR']
    args.batch_size=TP['Batch_Size']
    args.loss=TP['Loss']
    args.workers=TP['Workers']
    args.num_epochs=TP['Epochs']
    args.l2=TP['L2']
    args.brightness_scale = TP['brightness_scale']

    data_list = parse_datalist(args.data_root)
    if len(data_list) == 1:
        args.data_root = data_list[0]
    elif len(data_list) == 2:
        data_root_1, data_root_2 = data_list[0], data_list[1]
        if "OpenEDS" in data_root_1 and "Calipso" in data_root_2:
            pass
        elif "OpenEDS" in data_root_2 and "Calipso" in data_root_1:
            data_root_1, data_root_2 = data_root_2, data_root_1
        else:
            raise ValueError("Invalid data root {} and {}!".format(
                data_root_1,
                data_root_2,
            ))
        args.openeds_root, args.calipso_root = data_root_1, data_root_2
    else:
        raise ValueError("Invalid data root {}".format(
            data_list,
        ))

    return args
